//! MAI FORK (session 28): first-class memory-pool ownership — additive
//! only, value-neutral for every existing path. A [`CudaPool`] owns a
//! `CUmemoryPool` created with a configurable release threshold, so a
//! consumer (e.g. a session allocator) can isolate its allocations from
//! the device-default pool instead of sharing first-fit behavior across
//! streams. Allocations come back as OWNING [`CudaSlice`]s built
//! in-crate (the fields are `pub(crate)` by design; only this module
//! may construct them from raw pool handles).

use super::{CudaContext, CudaSlice, CudaStream};
use crate::driver::{result, sys, DriverError};
use std::sync::Arc;

/// An owned `CUmemoryPool` bound to its context.
pub struct CudaPool {
    pool: sys::CUmemoryPool,
    ctx: Arc<CudaContext>,
}

impl CudaPool {
    /// Create a pool on this context. `release_threshold` sets
    /// `CU_MEMPOOL_ATTR_RELEASE_THRESHOLD`: freed physical pages return
    /// to the OS once reserved exceeds the threshold (the driver default
    /// is `usize::MAX` = never release).
    pub fn new(ctx: &Arc<CudaContext>, release_threshold: Option<usize>) -> Result<Self, DriverError> {
        ctx.bind_to_thread()?;
        let mut props: sys::CUmemPoolProps = unsafe { std::mem::zeroed() };
        props.allocType = sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
        props.location = sys::CUmemLocation {
            type_: sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
            id: ctx.ordinal() as i32,
        };
        let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
        // SAFETY: props is fully initialized above; out-pointer pattern.
        unsafe { sys::cuMemPoolCreate(&mut pool, &mut props) }.result()?;
        let pool = CudaPool { pool, ctx: ctx.clone() };
        if let Some(bytes) = release_threshold {
            pool.set_release_threshold(bytes)?;
        }
        Ok(pool)
    }

    /// The raw handle (for result::mem_pool::* calls this module does not
    /// yet wrap).
    pub fn raw(&self) -> sys::CUmemoryPool {
        self.pool
    }

    pub fn set_release_threshold(&self, bytes: usize) -> Result<(), DriverError> {
        // SAFETY: plain attribute set on a live pool.
        unsafe {
            sys::cuMemPoolSetAttribute(
                self.pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                &bytes as *const usize as *mut _,
            )
        }
        .result()
    }

    pub fn release_threshold(&self) -> Result<usize, DriverError> {
        let mut v = 0usize;
        // SAFETY: plain attribute get on a live pool.
        unsafe {
            sys::cuMemPoolGetAttribute(
                self.pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                &mut v as *mut usize as *mut _,
            )
        }
        .result()?;
        Ok(v)
    }

    /// `cuMemPoolTrimTo` — ask the pool to return free physical memory to
    /// the OS down to `min_bytes_to_keep`.
    pub fn trim_to(&self, min_bytes_to_keep: usize) -> Result<(), DriverError> {
        // SAFETY: plain trim on a live pool.
        unsafe { sys::cuMemPoolTrimTo(self.pool, min_bytes_to_keep) }.result()
    }

    /// Stream-ordered allocation FROM THIS POOL as an OWNING
    /// [`CudaSlice`] (uninitialized memory). This is the piece upstream
    /// cudarc lacks: `CudaSlice` fields are `pub(crate)`, so a consumer
    /// allocator could not own a slice drawn from its own pool.
    ///
    /// # Safety
    /// The returned memory is uninitialized.
    pub unsafe fn alloc_async<T>(
        &self,
        stream: &Arc<CudaStream>,
        len: usize,
    ) -> Result<CudaSlice<T>, DriverError> {
        stream.ctx.bind_to_thread()?;
        let cu_device_ptr = result::mem_pool::alloc_async(
            self.pool,
            len * std::mem::size_of::<T>(),
            stream.cu_stream,
        )?;
        let (read, write) = if stream.ctx().is_managing_stream_synchronization() {
            (
                Some(stream.ctx().new_event(None)?),
                Some(stream.ctx().new_event(None)?),
            )
        } else {
            (None, None)
        };
        Ok(CudaSlice {
            cu_device_ptr,
            len,
            read,
            write,
            stream: stream.clone(),
            marker: std::marker::PhantomData,
        })
    }
}

impl Drop for CudaPool {
    fn drop(&mut self) {
        // Deliberately NOT destroyed: live allocations from this pool
        // would be invalidated. Pools are per-context and tiny; the
        // handle lives with the context.
        let _ = unsafe { sys::cuMemPoolTrimTo(self.pool, 0) };
    }
}
