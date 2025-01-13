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


    let positions = [
    (10, 35), (10, 36), (10, 37), (10, 38), (10, 39), (10, 40), (10, 41), (10, 42),
    (10, 43), (10, 44), (10, 45), (10, 46), (10, 47), (11, 0), (11, 1), (11, 2),
    (11, 3), (11, 4), (11, 5), (11, 6), (11, 7)
];

let color = 0xFDFBEC; // Color 253, 255, 238 in RGB Hex format


for (x, y) in positions.iter() {
    // offset
    let offset = (*y * width + *x) * 4;
    
    // write color to frame buffer
    buf.write_val(offset as usize, &color).expect("error writing frame buffer");
}


    // flush to screen
    device.flush().expect("failed to flush");
    early_println!("flushed to screen");
}