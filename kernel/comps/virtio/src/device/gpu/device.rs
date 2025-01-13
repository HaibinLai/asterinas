use alloc::{boxed::Box, sync::Arc, vec};
use core::hint::spin_loop;

use log::info;
use ostd::{
    early_println,
    mm::{DmaDirection, DmaStream, DmaStreamSlice, FrameAllocOptions, HasPaddr, VmIo},
    sync::SpinLock,
    trap::TrapFrame,
};

use super::{
    config::{GPUFeatures, VirtioGPUConfig},
    control::{
        VirtioGpuFormat, VirtioGpuMemEntry, VirtioGpuRect, VirtioGpuResourceAttachBacking, VirtioGpuResourceCreate2D, VirtioGpuResourceFlush, VirtioGpuRespAttachBacking, VirtioGpuRespDisplayInfo, VirtioGpuRespResourceFlush, VirtioGpuRespSetScanout, VirtioGpuRespTransferToHost2D, VirtioGpuRespUpdateCursor, VirtioGpuSetScanout, VirtioGpuTransferToHost2D, VirtioGpuUpdateCursor
    },
    header::VirtioGpuCtrlHdr,
};
use crate::{
    device::{
        gpu::{
            control::{
                VirtioGpuCursorPos, VirtioGpuGetEdid, VirtioGpuRespEdid, VirtioGpuRespResourceCreate2D, RESPONSE_SIZE
            },
            header::{VirtioGpuCtrlType, REQUEST_SIZE},
        },
        VirtioDeviceError,
    },
    queue::VirtQueue,
    transport::{ConfigManager, VirtioTransport},
};

/// Both virtqueues have the same format.
/// Each request and each response have a fixed header, followed by command specific data fields. See header.rs for the header format.
pub struct GPUDevice {
    config_manager: ConfigManager<VirtioGPUConfig>,

    /// queue for sending control commands
    control_queue: SpinLock<VirtQueue>,
    /// queue for sending cursor updates.
    /// According to virtio spec v1.3, 5.7.2 Virtqueues:
    /// The separate cursor queue is the "fast track" for cursor commands (VIRTIO_GPU_CMD_UPDATE_CURSOR and VIRTIO_GPU_CMD_MOVE_CURSOR),
    /// so they go through without being delayed by time-consuming commands in the control queue.
    cursor_queue: SpinLock<VirtQueue>,

    // request and response DMA buffer for control queue
    control_request: DmaStream,
    control_response: DmaStream,

    // request and response DMA buffer for cursor queue
    cursor_request: DmaStream,
    cursor_response: DmaStream,

    // Since the virtio gpu header remains consistent for both requests and responses,
    // we store it to avoid recreating the header repeatedly.
    header: VirtioGpuCtrlHdr,
    transport: SpinLock<Box<dyn VirtioTransport>>,
}

impl GPUDevice {
    const QUEUE_SIZE: u16 = 64;

    pub fn negotiate_features(features: u64) -> u64 {
        let features = GPUFeatures::from_bits_truncate(features);
        early_println!("virtio_gpu_features = {:?}", features);
        features.bits()
    }

    pub fn init(mut transport: Box<dyn VirtioTransport>) -> Result<(), VirtioDeviceError> {
        let config_manager = VirtioGPUConfig::new_manager(transport.as_ref());
        // TODO: read features and save as a field of device
        early_println!("virtio_gpu_config = {:?}", config_manager.read_config());

        // Initalize virtqueues
        const CONTROL_QUEUE_INDEX: u16 = 0;
        const CURSOR_QUEUE_INDEX: u16 = 1;
        // TODO(Taojie): the size of queues?
        let control_queue = SpinLock::new(
            VirtQueue::new(CONTROL_QUEUE_INDEX, Self::QUEUE_SIZE, transport.as_mut())
                .expect("create control queue failed"),
        );
        let cursor_queue = SpinLock::new(
            VirtQueue::new(CURSOR_QUEUE_INDEX, Self::QUEUE_SIZE, transport.as_mut())
                .expect("create cursor queue failed"),
        );

        // Initalize DMA buffers
        let control_request = {
            let vm_segment = FrameAllocOptions::new().alloc_segment(1).unwrap();
            DmaStream::map(vm_segment.into(), DmaDirection::Bidirectional, false).unwrap()
        };
        let control_response = {
            let vm_segment = FrameAllocOptions::new().alloc_segment(1).unwrap();
            DmaStream::map(vm_segment.into(), DmaDirection::Bidirectional, false).unwrap()
        };
        let cursor_request = {
            let vm_segment = FrameAllocOptions::new().alloc_segment(1).unwrap();
            DmaStream::map(vm_segment.into(), DmaDirection::Bidirectional, false).unwrap()
        };
        let cursor_response = {
            let vm_segment = FrameAllocOptions::new().alloc_segment(1).unwrap();
            DmaStream::map(vm_segment.into(), DmaDirection::Bidirectional, false).unwrap()
        };

        // Create device
        let device = Arc::new(Self {
            config_manager,
            control_queue,
            cursor_queue,
            control_request,
            control_response,
            cursor_request,
            cursor_response,
            header: VirtioGpuCtrlHdr::default(),
            transport: SpinLock::new(transport),
        });

        // Interrupt handler
        let clone_device = device.clone();
        let handle_irq_ctl = move |_: &TrapFrame| {
            clone_device.handle_irq();
        };
        let clone_device = device.clone();
        let handle_irq_cursor = move |_: &TrapFrame| {
            clone_device.handle_irq();
        };

        let clone_device = device.clone();
        let handle_config_change = move |_: &TrapFrame| {
            clone_device.handle_config_change();
        };

        // Register irq callbacks
        let mut transport = device.transport.lock();
        transport
            .register_queue_callback(CONTROL_QUEUE_INDEX, Box::new(handle_irq_ctl), false)
            .unwrap();
        transport
            .register_queue_callback(CURSOR_QUEUE_INDEX, Box::new(handle_irq_cursor), false)
            .unwrap();
        transport
            .register_cfg_callback(Box::new(handle_config_change))
            .unwrap();

        transport.finish_init();

        // Done: query the display information from the device using the VIRTIO_GPU_CMD_GET_DISPLAY_INFO command,
        //      and use that information for the initial scanout setup.

        // TODO: (Taojie) fetch the EDID information using the VIRTIO_GPU_CMD_GET_EDID command,
        //      If no information is available or all displays are disabled the driver MAY choose to use a fallback, such as 1024x768 at display 0.

        // TODO: (Taojie) query all shared memory regions supported by the device.
        //      If the device supports shared memory, the shmid of a region MUST be one of:
        //      - VIRTIO_GPU_SHM_ID_UNDEFINED  = 0
        //      - VIRTIO_GPU_SHM_ID_HOST_VISIBLE = 1
        // Taojie: I think the above requirement is too complex to implement.

        // Taojie: we directly test gpu functionality here rather than writing a user application.
        // Test device
        // test_frame_buffer(Arc::clone(&device));
        init_frame_buffer(Arc::clone(&device));
        // test_cursor(Arc::clone(&device));
        Ok(())
    }

    fn handle_config_change(&self) {
        info!("virtio_gpu: config space change");
    }

    fn handle_irq(&self) {
        info!("virtio_gpu handle irq");
        // TODO: follow the implementation of virtio_block
    }

    /// Retrieve the EDID data for a given scanout.
    ///  
    /// - Request data is struct virtio_gpu_get_edid).
    /// - Response type is VIRTIO_GPU_RESP_OK_EDID, response data is struct virtio_gpu_resp_edid.
    ///
    /// Support is optional and negotiated using the VIRTIO_GPU_F_EDID feature flag.
    /// The response contains the EDID display data blob (as specified by VESA) for the scanout.
    fn request_edid_info(&self) -> Result<(), VirtioDeviceError> {
        // Prepare request header DMA buffer
        // let request_header_slice = {
        //     let req_slice = DmaStreamSlice::new(&self.control_request, 0, size_of::<VirtioGpuCtrlHdr>());
        //     let req = VirtioGpuCtrlHdr {
        //         type_: VirtioGpuCtrlType::VIRTIO_GPU_CMD_GET_EDID as u32,
        //         ..VirtioGpuCtrlHdr::default()
        //     };
        //     req_slice.write_val(0, &req).unwrap();
        //     req_slice.sync().unwrap();
        //     req_slice
        // };

        // Prepare request data DMA buffer
        let request_data_slice = {
            let request_data_slice =
                DmaStreamSlice::new(&self.control_request, 0, size_of::<VirtioGpuGetEdid>());
            let req_data = VirtioGpuGetEdid::default();
            request_data_slice.write_val(0, &req_data).unwrap();
            request_data_slice.sync().unwrap();
            request_data_slice
        };

        let inputs = vec![&request_data_slice];

        // Prepare response DMA buffer
        let resp_slice = {
            let resp_slice =
                DmaStreamSlice::new(&self.control_response, 0, size_of::<VirtioGpuRespEdid>()); // TODO: response size
            resp_slice
                .write_val(0, &VirtioGpuRespEdid::default())
                .unwrap();
            resp_slice.sync().unwrap();
            resp_slice
        };

        // Add buffer to queue
        let mut control_queue = self.control_queue.disable_irq().lock();
        control_queue
            .add_dma_buf(inputs.as_slice(), &[&resp_slice])
            .expect("Add buffers to queue failed");

        // Notify
        if control_queue.should_notify() {
            control_queue.notify();
        }

        // Wait for response
        while !control_queue.can_pop() {
            spin_loop();
        }
        control_queue.pop_used().expect("Pop used failed");

        resp_slice.sync().unwrap();
        let resp: VirtioGpuRespEdid = resp_slice.read_val(0).unwrap();

        // type check
        if resp.header_type() != VirtioGpuCtrlType::VIRTIO_GPU_RESP_OK_EDID as u32 {
            return Err(VirtioDeviceError::QueueUnknownError);
        }

        early_println!("EDID info from virt_gpu device: {:?}", resp);

        Ok(())
    }

    fn resolution(&self) -> Result<(u32, u32), VirtioDeviceError> {
        let display_info = self.request_display_info()?;
        let rect = display_info.get_rect(0).unwrap();
        Ok((rect.width(), rect.height()))
    }

    fn request_display_info(&self) -> Result<VirtioGpuRespDisplayInfo, VirtioDeviceError> {
        // Prepare request DMA buffer
        let req_slice = {
            let req_slice = DmaStreamSlice::new(&self.control_request, 0, REQUEST_SIZE);
            let req = VirtioGpuCtrlHdr {
                type_: VirtioGpuCtrlType::VIRTIO_GPU_CMD_GET_DISPLAY_INFO as u32,
                ..VirtioGpuCtrlHdr::default()
            };
            req_slice.write_val(0, &req).unwrap();
            req_slice.sync().unwrap();
            req_slice
        };

        // Prepare response DMA buffer
        let resp_slice = {
            let resp_slice = DmaStreamSlice::new(&self.control_response, 0, RESPONSE_SIZE);
            resp_slice
                .write_val(0, &VirtioGpuRespDisplayInfo::default())
                .unwrap();
            resp_slice.sync().unwrap();
            resp_slice
        };

        // Add buffer to queue
        let mut control_queue = self.control_queue.disable_irq().lock();
        control_queue
            .add_dma_buf(&[&req_slice], &[&resp_slice])
            .expect("Add buffers to queue failed");

        // Notify
        if control_queue.should_notify() {
            control_queue.notify();
        }

        // Wait for response
        while !control_queue.can_pop() {
            // early_println!("waiting for response...");
            spin_loop();
        }
        control_queue.pop_used().expect("Pop used failed");

        resp_slice.sync().unwrap();
        let resp: VirtioGpuRespDisplayInfo = resp_slice.read_val(0).unwrap();
        // early_println!("display info from virt_gpu device: {:?}", resp);
        Ok(resp)
    }

    /// From the spec:
    ///
    /// Create a host resource using VIRTIO_GPU_CMD_RESOURCE_CREATE_2D.
    /// Allocate a framebuffer from guest ram, and attach it as backing storage to the resource just created, using VIRTIO_GPU_CMD_RESOURCE_ATTACH_BACKING.
    /// Use VIRTIO_GPU_CMD_SET_SCANOUT to link the framebuffer to a display scanout.
    ///
    /// Response type is VIRTIO_GPU_RESP_OK_NODATA.
    /// This creates a 2D resource on the host with the specified width, height and format. The resource ids are generated by the guest.
    fn resource_create_2d(
        &self,
        resource_id: u32,
        width: u32,
        height: u32,
    ) -> Result<(), VirtioDeviceError> {
        // Prepare request data DMA buffer
        let req_data_slice = {
            let req_data_slice = DmaStreamSlice::new(
                &self.control_request,
                0,
                size_of::<VirtioGpuResourceCreate2D>(),
            );
            early_println!(
                "parameters: resource_id: {}, width: {}, height: {}",
                resource_id,
                width,
                height
            );
            let req_data = VirtioGpuResourceCreate2D::new(
                resource_id,
                VirtioGpuFormat::VIRTIO_GPU_FORMAT_B8G8R8A8_UNORM,
                width,
                height,
            );
            req_data_slice.write_val(0, &req_data).unwrap();
            req_data_slice.sync().unwrap();
            req_data_slice
        };

        let inputs = vec![&req_data_slice];

        // Prepare response DMA buffer
        let resp_slice = {
            let resp_slice = DmaStreamSlice::new(
                &self.control_response,
                0,
                size_of::<VirtioGpuRespResourceCreate2D>(),
            );
            resp_slice
                .write_val(0, &VirtioGpuRespResourceCreate2D::default())
                .unwrap();
            resp_slice.sync().unwrap();
            resp_slice
        };

        // Add buffer to queue
        let mut control_queue = self.control_queue.disable_irq().lock();
        control_queue
            .add_dma_buf(inputs.as_slice(), &[&resp_slice])
            .expect("Add buffers to queue failed");

        // Notify
        if control_queue.should_notify() {
            control_queue.notify();
        }

        // Wait for response
        while !control_queue.can_pop() {
            spin_loop();
        }
        control_queue.pop_used().expect("Pop used failed");

        resp_slice.sync().unwrap();
        let resp: VirtioGpuRespResourceCreate2D = resp_slice.read_val(0).unwrap();

        // check response with type OK_NODATA
        if resp.header_type() != VirtioGpuCtrlType::VIRTIO_GPU_RESP_OK_NODATA as u32 {
            return Err(VirtioDeviceError::QueueUnknownError);
        }
        Ok(())
    }

    pub fn setup_framebuffer(&self) -> Result<Arc<DmaStream>, VirtioDeviceError> {
        // get display info
        let display_info = self.request_display_info()?;
        let rect = display_info.get_rect(0).unwrap();

        // create resource 2d
        self.resource_create_2d(0xbabe, rect.width(), rect.height())?;

        // alloc continuous memory for framebuffer
        // Each pixel is 4 bytes (32 bits) in RGBA format.
        let size = rect.width() as usize * rect.height() as usize * 4;
        let fracme_num = size / 4096 + 1; // TODO: (Taojie) use Asterinas API to represent page size.
        let frame_buffer_dma = {
            let vm_segment = FrameAllocOptions::new().alloc_segment(fracme_num).unwrap();
            DmaStream::map(vm_segment.into(), DmaDirection::ToDevice, false).unwrap()
        };

        // attach backing storage
        // TODO: (Taojie) excapsulate 0xbabe
        self.resource_attch_backing(0xbabe, frame_buffer_dma.paddr(), size as u32)?;

        // map frame buffer to screen
        self.set_scanout(rect, 0, 0xbabe)?;

        // return dma to be written
        Ok(Arc::new(frame_buffer_dma))
    }

    fn resource_attch_backing(
        &self,
        resource_id: i32,
        paddr: usize,
        size: u32,
    ) -> Result<(), VirtioDeviceError> {
        // Prepare request data DMA buffer
        let req_data_slice = {
            let req_data_slice = DmaStreamSlice::new(
                &self.control_request,
                0,
                size_of::<VirtioGpuResourceAttachBacking>(),
            );
            let req_data = VirtioGpuResourceAttachBacking::new(resource_id as u32, 1);
            req_data_slice.write_val(0, &req_data).unwrap();
            req_data_slice.sync().unwrap();
            req_data_slice
        };

        // Prepare request data DMA buffer
        let mem_entry_slice = {
            let mem_entry_slice = DmaStreamSlice::new(
                &self.control_request,
                size_of::<VirtioGpuResourceAttachBacking>(),
                size_of::<VirtioGpuMemEntry>(),
            );
            let mem_entry = VirtioGpuMemEntry::new(paddr, size);
            mem_entry_slice.write_val(0, &mem_entry).unwrap();
            mem_entry_slice.sync().unwrap();
            mem_entry_slice
        };

        let inputs = vec![&req_data_slice, &mem_entry_slice];

        // Prepare response DMA buffer
        let resp_slice = {
            let resp_slice = DmaStreamSlice::new(
                &self.control_response,
                0,
                size_of::<VirtioGpuRespAttachBacking>(),
            );
            resp_slice
                .write_val(0, &VirtioGpuRespAttachBacking::default())
                .unwrap();
            resp_slice.sync().unwrap();
            resp_slice
        };

        // Add buffer to queue
        let mut control_queue = self.control_queue.disable_irq().lock();
        control_queue
            .add_dma_buf(inputs.as_slice(), &[&resp_slice])
            .expect("Add buffers to queue failed");

        // Notify
        if control_queue.should_notify() {
            control_queue.notify();
        }

        // Wait for response
        while !control_queue.can_pop() {
            spin_loop();
        }
        control_queue.pop_used().expect("Pop used failed");

        resp_slice.sync().unwrap();
        let resp: VirtioGpuRespAttachBacking = resp_slice.read_val(0).unwrap();

        // check response with type OK_NODATA
        if resp.header_type() != VirtioGpuCtrlType::VIRTIO_GPU_RESP_OK_NODATA as u32 {
            return Err(VirtioDeviceError::QueueUnknownError);
        }

        Ok(())
    }

    fn set_scanout(
        &self,
        rect: VirtioGpuRect,
        scanout_id: i32,
        resource_id: i32,
    ) -> Result<(), VirtioDeviceError> {
        // Prepare request data DMA buffer
        let req_data_slice = {
            let req_data_slice =
                DmaStreamSlice::new(&self.control_request, 0, size_of::<VirtioGpuSetScanout>());
            let req_data = VirtioGpuSetScanout::new(scanout_id as u32, resource_id as u32, rect);
            req_data_slice.write_val(0, &req_data).unwrap();
            req_data_slice.sync().unwrap();
            req_data_slice
        };

        // Prepare response DMA buffer
        let resp_slice = {
            let resp_slice = DmaStreamSlice::new(
                &self.control_response,
                0,
                size_of::<VirtioGpuRespSetScanout>(),
            );
            resp_slice
                .write_val(0, &VirtioGpuRespSetScanout::default())
                .unwrap();
            resp_slice.sync().unwrap();
            resp_slice
        };

        // Add buffer to queue
        let mut control_queue = self.control_queue.disable_irq().lock();
        control_queue
            .add_dma_buf(&[&req_data_slice], &[&resp_slice])
            .expect("Add buffers to queue failed");

        // Notify
        if control_queue.should_notify() {
            control_queue.notify();
        }

        // Wait for response
        while !control_queue.can_pop() {
            spin_loop();
        }
        control_queue.pop_used().expect("Pop used failed");

        resp_slice.sync().unwrap();
        let resp: VirtioGpuRespSetScanout = resp_slice.read_val(0).unwrap();

        // check response with type OK_NODATA
        if resp.header_type() != VirtioGpuCtrlType::VIRTIO_GPU_RESP_OK_NODATA as u32 {
            return Err(VirtioDeviceError::QueueUnknownError);
        }

        Ok(())
    }

    pub fn flush(&self) -> Result<(), VirtioDeviceError> {
        // get rect info
        let display_info = self.request_display_info()?;
        let rect = display_info.get_rect(0).unwrap();

        // transfer from guest memmory to host resource
        self.transfer_to_host_2d(rect, 0, 0xbabe)?;

        // resource flush
        self.resource_flush(rect, 0xbabe)?;
        Ok(())
    }

    fn transfer_to_host_2d(
        &self,
        rect: VirtioGpuRect,
        offset: i32,
        resource_id: i32,
    ) -> Result<(), VirtioDeviceError> {
        // Prepare request data DMA buffer
        let req_data_slice = {
            let req_data_slice = DmaStreamSlice::new(
                &self.control_request,
                0,
                size_of::<VirtioGpuTransferToHost2D>(),
            );
            let req_data = VirtioGpuTransferToHost2D::new(rect, offset as u64, resource_id as u32);
            req_data_slice.write_val(0, &req_data).unwrap();
            req_data_slice.sync().unwrap();
            req_data_slice
        };

        // Prepare response DMA buffer
        let resp_slice = {
            let resp_slice = DmaStreamSlice::new(
                &self.control_response,
                0,
                size_of::<VirtioGpuRespTransferToHost2D>(),
            );
            resp_slice
                .write_val(0, &VirtioGpuRespTransferToHost2D::default())
                .unwrap();
            resp_slice.sync().unwrap();
            resp_slice
        };

        // Add buffer to queue
        let mut control_queue = self.control_queue.disable_irq().lock();
        control_queue
            .add_dma_buf(&[&req_data_slice], &[&resp_slice])
            .expect("Add buffers to queue failed");

        // Notify
        if control_queue.should_notify() {
            control_queue.notify();
        }

        // Wait for response
        while !control_queue.can_pop() {
            spin_loop();
        }
        control_queue.pop_used().expect("Pop used failed");

        resp_slice.sync().unwrap();
        let resp: VirtioGpuRespSetScanout = resp_slice.read_val(0).unwrap();

        // check response with type OK_NODATA
        if resp.header_type() != VirtioGpuCtrlType::VIRTIO_GPU_RESP_OK_NODATA as u32 {
            return Err(VirtioDeviceError::QueueUnknownError);
        }

        Ok(())
    }
    
    fn resource_flush(&self, rect: VirtioGpuRect, resource_id: i32) -> Result<(), VirtioDeviceError> {
        // Prepare request data DMA buffer
        let req_data_slice = {
            let req_data_slice = DmaStreamSlice::new(
                &self.control_request,
                0,
                size_of::<VirtioGpuResourceFlush>(),
            );
            let req_data = VirtioGpuResourceFlush::new(rect, resource_id as u32);
            req_data_slice.write_val(0, &req_data).unwrap();
            req_data_slice.sync().unwrap();
            req_data_slice
        };

        // Prepare response DMA buffer
        let resp_slice = {
            let resp_slice = DmaStreamSlice::new(
                &self.control_response,
                0,
                size_of::<VirtioGpuRespResourceFlush>(),
            );
            resp_slice
                .write_val(0, &VirtioGpuRespResourceFlush::default())
                .unwrap();
            resp_slice.sync().unwrap();
            resp_slice
        };

        // Add buffer to queue
        let mut control_queue = self.control_queue.disable_irq().lock();
        control_queue
            .add_dma_buf(&[&req_data_slice], &[&resp_slice])
            .expect("Add buffers to queue failed");

        // Notify
        if control_queue.should_notify() {
            control_queue.notify();
        }

        // Wait for response
        while !control_queue.can_pop() {
            spin_loop();
        }
        control_queue.pop_used().expect("Pop used failed");

        resp_slice.sync().unwrap();
        let resp: VirtioGpuRespSetScanout = resp_slice.read_val(0).unwrap();

        // check response with type OK_NODATA
        if resp.header_type() != VirtioGpuCtrlType::VIRTIO_GPU_RESP_OK_NODATA as u32 {
            return Err(VirtioDeviceError::QueueUnknownError);
        }
        Ok(())
    }

    pub fn update_cursor(&self, resource_id: u32, scanout_id: u32, pos_x: u32, pos_y: u32, hot_x: u32, hot_y: u32, move_only: bool) -> Result<(), VirtioDeviceError> {
        // Prepare request data DMA buffer
        // TODO: (Taojie) implement move cursor onlys
        let req_data_slice = {
            let req_data_slice = DmaStreamSlice::new(
                &self.cursor_request,
                0,
                size_of::<VirtioGpuUpdateCursor>(),
            );
            let cursor_pos = VirtioGpuCursorPos::new(scanout_id, 0, 0);
            let req_data = VirtioGpuUpdateCursor::new(cursor_pos, 0xdade, 32, 32);
            req_data_slice.write_val(0, &req_data).unwrap();
            req_data_slice.sync().unwrap();
            req_data_slice
        };

        // Prepare response DMA buffer
        let resp_slice: DmaStreamSlice<&DmaStream> = {
            let resp_slice = DmaStreamSlice::new(
                &self.cursor_response,
                0,
                size_of::<VirtioGpuRespUpdateCursor>(),
            );
            resp_slice
                .write_val(0, &VirtioGpuRespUpdateCursor::default())
                .unwrap();
            resp_slice.sync().unwrap();
            resp_slice
        };

        // Add buffer to queue
        let mut cursor_queue = self.cursor_queue.disable_irq().lock();
        cursor_queue
            .add_dma_buf(&[&req_data_slice], &[&resp_slice])
            .expect("Add buffers to queue failed");

        // Notify
        if cursor_queue.should_notify() {
            cursor_queue.notify();
        }

        // Wait for response
        while !cursor_queue.can_pop() {
            spin_loop();
        }
        cursor_queue.pop_used().expect("Pop used failed");

        resp_slice.sync().unwrap();
        let resp: VirtioGpuRespUpdateCursor = resp_slice.read_val(0).unwrap();

        // check response with type OK_NODATA
        early_println!("update cursor response: {:?}", resp);
        if resp.header_type() != VirtioGpuCtrlType::VIRTIO_GPU_RESP_OK_NODATA as u32 {
            return Err(VirtioDeviceError::QueueUnknownError);
        }

        Ok(())
    }
}

/// Test the functionality of rendering cursor.
fn test_cursor(device: Arc<GPUDevice>) {
    // setup cursor
    // from spec: The mouse cursor image is a normal resource, except that it must be 64x64 in size. 
    // The driver MUST create and populate the resource (using the usual VIRTIO_GPU_CMD_RESOURCE_CREATE_2D, VIRTIO_GPU_CMD_RESOURCE_ATTACH_BACKING and VIRTIO_GPU_CMD_TRANSFER_TO_HOST_2D controlq commands) 
    // and make sure they are completed (using VIRTIO_GPU_FLAG_FENCE).
    let cursor_rect: VirtioGpuRect = VirtioGpuRect::new(0, 0, 64, 64);
    let size = cursor_rect.width() as usize * cursor_rect.height() as usize * 4;
    let cursor_dma_buffer = {
        let vm_segment = FrameAllocOptions::new().alloc_segment(size / 4096 + 1).unwrap();
        DmaStream::map(vm_segment.into(), DmaDirection::ToDevice, false).unwrap()
    };

    // write content into the cursor buffer: all black
    for y in 0..cursor_rect.height() {
        for x in 0..cursor_rect.width() {
            let offset = (y * cursor_rect.width() + x) * 4;
            let color = 0x00000000;
            cursor_dma_buffer.write_val(offset as usize, &color).unwrap();
        }
    }
    
    // create cursor resource, attach backing storage and transfer to host
    device.resource_create_2d(0xdade, cursor_rect.width(), cursor_rect.height()).unwrap();       // TODO: (Taojie) replace dade with cursor resource id, which is customized.
    device.resource_attch_backing(0xdade, cursor_dma_buffer.paddr(), size as u32).unwrap();
    device.transfer_to_host_2d(cursor_rect, 0, 0xdade).unwrap();

    early_println!("cursor setup done");
    // wait for some time 
    for _ in 0..1000000 {
    }

    // update current cursor
    device.update_cursor(0xdade, 0, 0, 0, 0, 0, false).unwrap();



}


/// Test the functionality of gpu device and driver.
fn test_frame_buffer(device: Arc<GPUDevice>) {
    // get resolution
    let (width, height) = device.resolution().expect("failed to get resolution");
    early_println!("[INFO] resolution: {}x{}", width, height);

    // test: get edid info
    device.request_edid_info().expect("failed to get edid info");

    // setup framebuffer
    let buf = device
        .setup_framebuffer()
        .expect("failed to setup framebuffer");

    // write content into buffer
    // for x in 0..height {
    //     for y in 0..width {
    //         let offset = (x * width + y) * 4;
    //         let color = if x % 2 == 0 && y % 2 == 0 {
    //             0x00ff_0000
    //         } else {
    //             0x0000_ff00
    //         };
    //         buf.write_val(offset as usize, &color).unwrap();
    //     }
    // }
    for y in 0..height {    //height=800
        for x in 0..width { //width=1280
            let offset = (y * width + x) * 4;
            // fb[idx] = x as u8;
            // fb[idx + 1] = y as u8;
            // fb[idx + 2] = (x + y) as u8;
            buf.write_val(offset as usize, &x).expect("error writing frame buffer");
            buf.write_val((offset + 1) as usize, &y).expect("error writing frame buffer");
            buf.write_val((offset + 2) as usize, &(x+y)).expect("error writing frame buffer");
        }
    }

    // flush to screen
    device.flush().expect("failed to flush");
    early_println!("flushed to screen");
}


/// Test the functionality of gpu device and driver.
fn init_frame_buffer(device: Arc<GPUDevice>) {
    // get resolution
    let (width, height) = device.resolution().expect("failed to get resolution");
    early_println!("[INFO] resolution: {}x{}", width, height);

    // test: get edid info
    device.request_edid_info().expect("failed to get edid info");

    // setup framebuffer
    let buf = device
        .setup_framebuffer()
        .expect("failed to setup framebuffer");

    // write content into buffer
    for x in 0..height {
        for y in 0..width {
            let offset = (x * width + y) * 4;
            let color = 0x0000_0000;
            buf.write_val(offset as usize, &color).unwrap();
        }
    }

    // draw Asterinas logo
    let positions =[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 12), (0, 13), (0, 14), (0, 15), (0, 16), (0, 17), (0, 18), (0, 19), (0, 20), (0, 21), (0, 22), (0, 23), (0, 24), (0, 25), (0, 26), (0, 27), (0, 28), (0, 29), (0, 30), 
    (0, 31), (0, 32), (0, 33), (0, 34), (0, 35), (0, 36), (0, 37), (0, 38), (0, 39), (0, 40), (0, 41), (0, 42), (0, 43), (0, 44), (0, 45), (0, 46), (0, 47), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (1, 14), (1, 15), (1, 16), (1, 17), (1, 18), (1, 19), (1, 20), (1, 21), (1, 22), (1, 23), (1, 24), (1, 25), (1, 26), (1, 27), (1, 28), (1, 29), (1, 30), (1, 31), (1, 32), (1, 33), (1, 34), (1, 35), (1, 36), (1, 37), (1, 38), (1, 39), (1, 40), (1, 41), (1, 42), (1, 43), (1, 44), (1, 45), (1, 46), (1, 47), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (2, 11), (2, 12), (2, 13), (2, 14), (2, 15), (2, 16), (2, 17), (2, 18), (2, 19), (2, 20), (2, 21), (2, 22), (2, 23), (2, 24), (2, 25), (2, 26), (2, 27), (2, 28), (2, 29), (2, 30), (2, 31), (2, 32), (2, 33), (2, 34), (2, 35), (2, 36), (2, 37), (2, 38), (2, 39), (2, 40), (2, 41), (2, 42), 
    (2, 43), (2, 44), (2, 45), (2, 46), (2, 47), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (3, 11), (3, 12), (3, 13), (3, 14), (3, 15), (3, 16), (3, 17), (3, 18), (3, 19), (3, 20), (3, 21), (3, 22), (3, 23), (3, 24), (3, 25), (3, 26), (3, 27), (3, 28), (3, 29), (3, 30), (3, 31), (3, 32), (3, 33), (3, 34), (3, 35), (3, 36), (3, 37), (3, 38), (3, 39), (3, 40), (3, 41), (3, 42), (3, 43), (3, 44), (3, 45), (3, 46), 
    (3, 47), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (4, 11), (4, 12), (4, 13), (4, 14), (4, 15), (4, 16), (4, 17), (4, 18), (4, 19), (4, 20), (4, 21), (4, 22), (4, 23), (4, 24), (4, 25), (4, 26), (4, 27), (4, 28), (4, 29), (4, 30), (4, 31), (4, 32), (4, 33), (4, 34), (4, 35), (4, 36), (4, 37), (4, 38), (4, 39), (4, 40), (4, 41), (4, 42), (4, 43), (4, 44), 
    (4, 45), (4, 46), (4, 47), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10), 
    (5, 11), (5, 12), (5, 13), (5, 14), (5, 15), (5, 16), (5, 17), (5, 18), (5, 19), (5, 20), (5, 21), (5, 22), (5, 23), (5, 24), (5, 25), (5, 26), (5, 27), (5, 28), (5, 29), (5, 30), (5, 31), (5, 32), (5, 33), (5, 34), (5, 35), (5, 36), (5, 37), (5, 38), (5, 39), (5, 40), (5, 41), (5, 42), (5, 43), (5, 44), (5, 45), (5, 46), (5, 47), (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), 
    (6, 7), (6, 8), (6, 9), (6, 10), (6, 11), (6, 12), (6, 13), (6, 14), (6, 15), (6, 16), (6, 17), (6, 18), (6, 19), (6, 20), (6, 21), (6, 22), (6, 23), (6, 24), (6, 25), (6, 26), (6, 27), (6, 28), (6, 29), (6, 30), (6, 31), (6, 32), (6, 33), (6, 34), (6, 35), (6, 36), (6, 37), (6, 38), (6, 39), (6, 40), (6, 41), (6, 42), (6, 43), (6, 44), (6, 45), (6, 46), (6, 47), (7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7), (7, 8), (7, 9), (7, 10), (7, 11), (7, 12), (7, 13), (7, 14), (7, 15), (7, 16), (7, 17), (7, 18), (7, 19), (7, 20), (7, 21), (7, 22), (7, 23), (7, 24), (7, 25), (7, 26), (7, 27), (7, 28), (7, 29), (7, 30), (7, 31), (7, 32), (7, 33), (7, 34), (7, 35), (7, 36), (7, 37), (7, 38), (7, 39), (7, 40), (7, 41), (7, 42), (7, 43), (7, 44), (7, 45), (7, 46), (7, 47), (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8), (8, 9), (8, 10), (8, 11), (8, 12), (8, 13), (8, 14), (8, 15), (8, 16), (8, 17), (8, 18), (8, 19), (8, 20), (8, 21), (8, 22), (8, 23), (8, 24), (8, 25), (8, 26), (8, 27), (8, 28), (8, 29), (8, 30), (8, 31), (8, 32), (8, 33), (8, 34), (8, 35), (8, 36), (8, 37), (8, 38), (8, 39), 
    (8, 40), (8, 41), (8, 42), (8, 43), (8, 44), (8, 45), (8, 46), (8, 47), (9, 0), (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9), (9, 10), (9, 11), (9, 12), (9, 13), (9, 14), (9, 15), (9, 16), (9, 17), (9, 18), (9, 19), (9, 20), (9, 21), (9, 22), (9, 23), (9, 24), (9, 25), (9, 26), (9, 27), (9, 28), (9, 29), (9, 30), (9, 31), (9, 32), (9, 33), (9, 34), (9, 35), (9, 36), (9, 37), (9, 38), (9, 39), (9, 40), (9, 41), (9, 42), (9, 43), (9, 44), (9, 45), (9, 46), (9, 47), (10, 0), (10, 1), (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10), (10, 11), (10, 12), (10, 13), (10, 14), (10, 15), (10, 16), (10, 17), (10, 18), (10, 19), (10, 20), (10, 21), (10, 22), (10, 23), (10, 24), (10, 25), (10, 26), (10, 27), (10, 28), (10, 29), (10, 30), (10, 31), (10, 32), (10, 33), (10, 34), (10, 35), (10, 36), (10, 37), (10, 38), (10, 39), (10, 40), (10, 41), (10, 42), (10, 43), (10, 44), (10, 45), (10, 46), (10, 47), (11, 0), (11, 1), (11, 2), (11, 3), (11, 4), (11, 5), (11, 6), (11, 7), (11, 8), (11, 9), (11, 10), (11, 11), (11, 12), (11, 13), (11, 14), (11, 15), (11, 16), (11, 17), (11, 18), (11, 19), (11, 20), (11, 21), (11, 22), (11, 23), (11, 24), (11, 25), (11, 26), (11, 27), (11, 28), (11, 29), (11, 30), (11, 31), (11, 32), (11, 33), (11, 34), (11, 35), (11, 36), (11, 37), (11, 38), (11, 39), (11, 40), (11, 41), (11, 42), (11, 43), (11, 44), (11, 45), 
    (11, 46), (11, 47)];


let color = 0xFDFBEC; // Color 253, 255, 238 in RGB Hex format


// for (x, y) in positions.iter() {
//     // offset
//     let offset = (x * width + y) * 4;
//     let offset2 = (x * width + x + y) * 4;
//     let offset3 = (x * width + x+ x + y) * 4;
//     let offset4 = (x * width + x+ x+ x + y) * 4;
    
//     // write color to frame buffer
//     buf.write_val(offset as usize, &color).expect("error writing frame buffer");
//     buf.write_val(offset2 as usize, &color).expect("error writing frame buffer");
//     buf.write_val(offset3 as usize, &color).expect("error writing frame buffer");
//     buf.write_val(offset4 as usize, &color).expect("error writing frame buffer");
// }


    // flush to screen
    device.flush().expect("failed to flush");
    early_println!("flushed to screen");
}