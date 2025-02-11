use criterion::{
    Criterion,
    criterion_main,
    criterion_group
};
use rayon::prelude::*;
use std::{slice, sync::Arc, vec};

// Tensor Implementation
pub struct Tensor<T> {
    data: Arc<Box<[T]>>,
    shape: Vec<usize>,
    offset: usize,
    length: usize,
}

impl<T: Copy + Clone + Default> Tensor<T> {
    pub fn new(data: Vec<T>, shape: &Vec<usize>) -> Self {
        let length = data.len();
        Tensor {
            data: Arc::new(data.into_boxed_slice().try_into().unwrap()),
            shape: shape.clone(),
            offset: 0,
            length,
        }
    }

    pub fn default(shape: &Vec<usize>) -> Self {
        let length = shape.iter().product();
        let data = vec![T::default(); length];
        Self::new(data, shape)
    }

    pub fn data(&self) -> &[T] {
        &self.data[self.offset..][..self.length]
    }

    pub unsafe fn data_mut(&mut self) -> &mut [T] {
        let ptr = self.data.as_ptr().add(self.offset) as *mut T;
        slice::from_raw_parts_mut(ptr, self.length)
    }

    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    pub fn size(&self) -> usize {
        self.length
    }

    // Reinterpret the tensor as a new shape while preserving total size.
    pub fn reshape(&mut self, new_shape: &Vec<usize>) -> &mut Self {
        let new_length: usize = new_shape.iter().product();
        if new_length != self.length {
            let old_shape = self.shape.clone();
            panic!("New shape {new_shape:?} does not match tensor of {old_shape:?}");
        }
        self.shape = new_shape.clone();
        self
    }

    pub fn slice(&self, start: usize, shape: &Vec<usize>) -> Self {
        let new_length: usize = shape.iter().product();
        assert!(self.offset + start + new_length <= self.length);
        Tensor {
            data: self.data.clone(),
            shape: shape.clone(),
            offset: self.offset + start,
            length: new_length,
        }
    }


}

// Some helper functions for testing and debugging
impl Tensor<f32> {
    #[allow(unused)]
    pub fn close_to(&self, other: &Self, rel: f32) -> bool {
        if self.shape() != other.shape() {
            return false;
        }
        let a = self.data();
        let b = other.data();
        
        return a.iter().zip(b).all(|(x, y)| float_eq(x, y, rel));
    }
    #[allow(unused)]
    pub fn print(&self){
        println!(
            "shpae: {:?}, offset: {}, length: {}",
            self.shape, self.offset, self.length
        );
        let dim = self.shape()[self.shape().len() - 1];
        let batch = self.length / dim;
        for i in 0..batch {
            let start = i * dim;
            println!("{:?}", &self.data()[start..][..dim]);
        }
    }
}

#[inline]
pub fn float_eq(x: &f32, y: &f32, rel: f32) -> bool {
    (x - y).abs() <= rel * (x.abs() + y.abs()) / 2.0
}

// RmsNorm
pub fn rms_norm(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<f32>, epsilon: f32) {
    // todo!("实现 rms_norm，计算前做一些必要的检查会帮助你后续调试")
    assert!(y.size() == x.size());

    let n = x.shape().last().copied().unwrap_or(0);
    let batch = x.size() / n;

    let y_data = unsafe { y.data_mut() };
    let x_data = x.data();
    let w_data = w.data();

    for i in 0..batch {
        let offset = i * n;

        // Rayon Parller
        let mut s: f32 = (offset..n+offset).into_par_iter()
            .map(|i| x_data[i].powi(2)).sum();

        s = (s / n as f32 + epsilon).sqrt();

        for j in 0..n {
            y_data[offset + j] = w_data[j] * x_data[offset + j] / s;
        }
    }
}

pub fn urms_norm(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<f32>, epsilon: f32) {
    // todo!("实现 rms_norm，计算前做一些必要的检查会帮助你后续调试")
    assert!(y.size() == x.size());

    let n = x.shape().last().copied().unwrap_or(0);
    let batch = x.size() / n;

    let y_data = unsafe { y.data_mut() };
    let x_data = x.data();
    let w_data = w.data();

    for i in 0..batch {
        let offset = i * n;

        // common
        let mut s: f32 = (offset..n+offset).into_iter()
            .map(|i| x_data[i].powi(2)).sum();

        s = (s / n as f32 + epsilon).sqrt();

        for j in 0..n {
            y_data[offset + j] = w_data[j] * x_data[offset + j] / s;
        }
    }
}
fn matmul_transb(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) {
    // todo!("实现 matmul_transb，计算前做一些必要的检查会帮助你后续调试");
    assert!(a.shape().len() == 2);
    assert!(b.shape().len() == 2);
    assert!(c.shape().len() == 2);

    let (a_r, a_c) = (a.shape()[0], a.shape()[1]);
    let (b_r, b_c) = (b.shape()[0], b.shape()[1]);
    let (c_r, c_c) = (c.shape()[0], c.shape()[1]);

    assert!(a_c == b_c);
    assert!(a_r == c_r && b_r == c_c);

    let _a = a.data();
    let _b = b.data();
    let _c = unsafe { c.data_mut() };

    // Rayon Parller
    let pool = rayon::ThreadPoolBuilder::new().num_threads(8).build().unwrap();

    pool.scope(|s| {
        for (a, c) in _a.chunks(a_c).zip(_c.chunks_mut(c_c)) {
            s.spawn(|_| {
                for (b, cc) in _b.chunks(b_c).zip(c.iter_mut()) {
                    let d = b.iter()
                        .zip(a.iter())
                        .fold(0.,|s, (aa, bb)| s + aa * bb);
                    *cc = beta * (*cc) + alpha * d;
                }
            })
        }
    })
}

fn umatmul_transb(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) {
    // todo!("实现 matmul_transb，计算前做一些必要的检查会帮助你后续调试");
    assert!(a.shape().len() == 2);
    assert!(b.shape().len() == 2);
    assert!(c.shape().len() == 2);

    let (a_r, a_c) = (a.shape()[0], a.shape()[1]);
    let (b_r, b_c) = (b.shape()[0], b.shape()[1]);
    let (c_r, c_c) = (c.shape()[0], c.shape()[1]);

    assert!(a_c == b_c);
    assert!(a_r == c_r && b_r == c_c);

    let _a = a.data();
    let _b = b.data();
    let _c = unsafe { c.data_mut() };

    for i in 0..c_r {
        for j in 0..c_c {
            let mut s = 0.0;
            for k in 0..a_c {
                s += _a[i * a_c + k] * _b[j * b_c + k];
            }
            _c[i * c_c + j] = beta * _c[i * c_c + j] + alpha * s;
        }
    }

}

fn swiglu(y: &mut Tensor<f32>, x: &Tensor<f32>) {
    let len = y.size();
    assert!(len == x.size());

    let _y = unsafe { y.data_mut() };
    let _x = x.data();

    // todo!("实现 silu，这里给了一些前期准备工作的提示，你可以参考")

    // Rayon Paraller
    _y.par_iter_mut()
        .zip(_x.par_iter())
        .for_each(|(y, x)| *y *= x / (1.0 + (-x).exp()));
}

fn uswiglu(y: &mut Tensor<f32>, x: &Tensor<f32>) {
    let len = y.size();
    assert!(len == x.size());

    let _y = unsafe { y.data_mut() };
    let _x = x.data();

    // todo!("实现 silu，这里给了一些前期准备工作的提示，你可以参考")

    for i in 0..len {
        _y[i] *= _x[i] / (1.0 + (-_x[i]).exp())
    }

}

fn bench_rms_norm(c: &mut Criterion) {
    let mut g = c.benchmark_group("RMSNorm");
    let mut y = Tensor::<f32>::new(
        vec![
        0.1, 2.1, 3.2, 1.0, 1.7, 2.9, 3.5, 6.1, 0.7, 7.2,
        0.1, 2.1, 3.2, 1.0, 1.7, 2.9, 3.5, 6.1, 0.7, 7.2,
        0.1, 2.1, 3.2, 1.0, 1.7, 2.9, 3.5, 6.1, 0.7, 7.2,
        0.1, 2.1, 3.2, 1.0, 1.7, 2.9, 3.5, 6.1, 0.7, 7.2,
        0.1, 2.1, 3.2, 1.0, 1.7, 2.9, 3.5, 6.1, 0.7, 7.2,
        0.1, 2.1, 3.2, 1.0, 1.7, 2.9, 3.5, 6.1, 0.7, 7.2,
        0.1, 2.1, 3.2, 1.0, 1.7, 2.9, 3.5, 6.1, 0.7, 7.2,
        0.1, 2.1, 3.2, 1.0, 1.7, 2.9, 3.5, 6.1, 0.7, 7.2,
        0.1, 2.1, 3.2, 1.0, 1.7, 2.9, 3.5, 6.1, 0.7, 7.2,
        0.1, 2.1, 3.2, 1.0, 1.7, 2.9, 3.5, 6.1, 0.7, 7.2,
        0.8, 2.8, 1.2, 1.0, 1.7, 2.9, 4.1, 8.1, 3.7, 7.2,
        0.8, 2.8, 1.2, 1.0, 1.7, 2.9, 4.1, 8.1, 3.7, 7.2,
        0.8, 2.8, 1.2, 1.0, 1.7, 2.9, 4.1, 8.1, 3.7, 7.2,
        0.8, 2.8, 1.2, 1.0, 1.7, 2.9, 4.1, 8.1, 3.7, 7.2,
        0.8, 2.8, 1.2, 1.0, 1.7, 2.9, 4.1, 8.1, 3.7, 7.2,
        0.8, 2.8, 1.2, 1.0, 1.7, 2.9, 4.1, 8.1, 3.7, 7.2,
        0.8, 2.8, 1.2, 1.0, 1.7, 2.9, 4.1, 8.1, 3.7, 7.2,
        0.8, 2.8, 1.2, 1.0, 1.7, 2.9, 4.1, 8.1, 3.7, 7.2,
        0.8, 2.8, 1.2, 1.0, 1.7, 2.9, 4.1, 8.1, 3.7, 7.2,
        0.8, 2.8, 1.2, 1.0, 1.7, 2.9, 4.1, 8.1, 3.7, 7.2,
        ],
        &vec![20, 10]
    );
    let x = Tensor::<f32>::new(
        vec![
        0.1, 2.1, 3.2, 1.0, 1.7, 2.9, 3.5, 6.1, 0.7, 7.2,
        0.1, 2.1, 3.2, 1.0, 1.7, 2.9, 3.5, 6.1, 0.7, 7.2,
        0.1, 2.1, 3.2, 1.0, 1.7, 2.9, 3.5, 6.1, 0.7, 7.2,
        0.1, 2.1, 3.2, 1.0, 1.7, 2.9, 3.5, 6.1, 0.7, 7.2,
        0.1, 2.1, 3.2, 1.0, 1.7, 2.9, 3.5, 6.1, 0.7, 7.2,
        0.1, 2.1, 3.2, 1.0, 1.7, 2.9, 3.5, 6.1, 0.7, 7.2,
        0.1, 2.1, 3.2, 1.0, 1.7, 2.9, 3.5, 6.1, 0.7, 7.2,
        0.1, 2.1, 3.2, 1.0, 1.7, 2.9, 3.5, 6.1, 0.7, 7.2,
        0.1, 2.1, 3.2, 1.0, 1.7, 2.9, 3.5, 6.1, 0.7, 7.2,
        0.1, 2.1, 3.2, 1.0, 1.7, 2.9, 3.5, 6.1, 0.7, 7.2,
        0.8, 2.8, 1.2, 1.0, 1.7, 2.9, 4.1, 8.1, 3.7, 7.2,
        0.8, 2.8, 1.2, 1.0, 1.7, 2.9, 4.1, 8.1, 3.7, 7.2,
        0.8, 2.8, 1.2, 1.0, 1.7, 2.9, 4.1, 8.1, 3.7, 7.2,
        0.8, 2.8, 1.2, 1.0, 1.7, 2.9, 4.1, 8.1, 3.7, 7.2,
        0.8, 2.8, 1.2, 1.0, 1.7, 2.9, 4.1, 8.1, 3.7, 7.2,
        0.8, 2.8, 1.2, 1.0, 1.7, 2.9, 4.1, 8.1, 3.7, 7.2,
        0.8, 2.8, 1.2, 1.0, 1.7, 2.9, 4.1, 8.1, 3.7, 7.2,
        0.8, 2.8, 1.2, 1.0, 1.7, 2.9, 4.1, 8.1, 3.7, 7.2,
        0.8, 2.8, 1.2, 1.0, 1.7, 2.9, 4.1, 8.1, 3.7, 7.2,
        0.8, 2.8, 1.2, 1.0, 1.7, 2.9, 4.1, 8.1, 3.7, 7.2,
        ],
        &vec![20, 10]
    );
    let w = Tensor::<f32>::new(
        vec![0.1, 2.7, 3.3, 2.4, 7.1, 1.8, 3.7, 9.1, 0.7, 1.1],
        &vec![10]
    );
    g.bench_function("Rayon",
        |b| b.iter(|| rms_norm(&mut y, &x, &w, 1e-6))
    );
    g.bench_function("Common",
        |b| b.iter(|| urms_norm(&mut y, &x, &w, 1e-6))
    );
    g.finish();
}

fn bench_matmul(c: &mut Criterion){
    let mut g = c.benchmark_group("matmul");
    let mut c = Tensor::<f32>::new(
        vec![
        0.1, 3.1, 3.2, 1.5, 3.7, 2.7, 3.5, 6.1, 0.7, 7.2,
        0.1, 3.1, 3.2, 1.5, 3.7, 2.7, 3.5, 6.1, 0.7, 7.2,
        0.1, 3.1, 3.2, 1.5, 3.7, 2.7, 3.5, 6.1, 0.7, 7.2,
        0.1, 3.1, 3.2, 1.5, 3.7, 2.7, 3.5, 6.1, 0.7, 7.2,
        0.1, 3.1, 3.2, 1.5, 3.7, 2.7, 3.5, 6.1, 0.7, 7.2,
        0.7, 3.8, 1.2, 1.5, 2.1, 2.7, 4.2, 8.1, 3.7, 7.2,
        0.8, 2.8, 1.2, 1.0, 3.7, 2.9, 4.1, 8.1, 3.7, 7.2,
        0.8, 2.8, 1.2, 1.0, 3.7, 2.9, 4.1, 8.1, 3.7, 7.2,
        0.2, 2.8, 1.2, 1.0, 3.7, 2.9, 4.4, 8.1, 3.7, 7.2,
        0.8, 2.8, 1.2, 1.0, 1.7, 2.9, 4.1, 8.1, 3.7, 7.2,
        ],
        &vec![10, 10]
    );
    let a = Tensor::<f32>::new(
        vec![
        0.1, 3.1, 3.2, 1.5, 3.7, 2.7, 3.5, 6.1, 0.7, 7.2,
        0.1, 3.1, 3.2, 1.5, 3.7, 2.7, 3.5, 6.1, 0.7, 7.2,
        0.1, 3.1, 3.2, 1.5, 3.7, 2.7, 3.5, 6.1, 0.7, 7.2,
        0.1, 3.1, 3.2, 1.5, 3.7, 2.7, 3.5, 6.1, 0.7, 7.2,
        0.1, 3.1, 3.2, 1.5, 3.7, 2.7, 3.5, 6.1, 0.7, 7.2,
        0.8, 3.8, 1.2, 1.5, 2.1, 2.7, 4.2, 8.1, 3.7, 7.2,
        0.8, 2.8, 1.2, 1.0, 3.7, 2.9, 4.1, 8.1, 3.7, 7.2,
        0.8, 2.8, 1.2, 1.0, 3.7, 2.9, 4.1, 8.1, 3.7, 7.2,
        0.8, 2.8, 1.2, 1.0, 3.7, 2.9, 4.4, 8.1, 3.7, 7.2,
        0.8, 2.8, 1.2, 1.0, 1.7, 2.9, 4.1, 8.1, 3.7, 7.2,
        ],
        &vec![10, 10]
    );
    let b = Tensor::<f32>::new(
        vec![
        0.1, 3.1, 3.2, 1.0, 3.7, 2.7, 3.5, 6.1, 0.7, 7.2,
        0.1, 3.1, 3.2, 1.0, 3.7, 2.7, 3.5, 6.1, 0.7, 7.2,
        0.1, 3.1, 3.2, 1.0, 3.7, 2.7, 3.5, 6.1, 0.7, 7.2,
        0.1, 3.1, 3.2, 1.0, 3.7, 2.7, 3.5, 6.1, 0.7, 7.2,
        0.1, 3.1, 3.2, 1.0, 3.7, 2.7, 3.5, 6.1, 0.7, 7.2,
        0.8, 3.8, 1.2, 1.0, 3.7, 2.7, 4.1, 8.1, 3.7, 7.2,
        0.8, 2.8, 1.2, 1.0, 3.7, 2.9, 4.1, 8.1, 3.7, 7.2,
        0.8, 2.8, 1.2, 1.0, 3.7, 2.9, 4.1, 8.1, 3.7, 7.2,
        0.8, 2.8, 1.2, 1.0, 3.7, 2.9, 4.1, 8.1, 3.7, 7.2,
        0.8, 2.8, 1.2, 1.0, 1.7, 2.9, 4.1, 8.1, 3.7, 7.2,
        ],
        &vec![10, 10]
    );
    g.bench_function("Rayon", 
        |d| d.iter(
            || matmul_transb(&mut c, 0.5, &a, &b, 1.)
        )
    );
    g.bench_function("Common", 
        |d| d.iter(
            || umatmul_transb(&mut c, 0.5, &a, &b, 1.)
        )
    );

    g.finish();
}

fn bench_swiglu(c: &mut Criterion){
    let mut g = c.benchmark_group("Swiglu");
    let mut y = Tensor::<f32>::new(
        vec![
        0.1, 2.1, 3.2, 1.0, 1.7, 2.9, 3.5, 6.1, 0.7, 7.2,
        0.1, 2.1, 3.2, 1.0, 1.7, 2.9, 3.5, 6.1, 0.7, 7.2,
        0.1, 2.1, 3.2, 1.0, 1.7, 2.9, 3.5, 6.1, 0.7, 7.2,
        0.1, 2.1, 3.2, 1.0, 1.7, 2.9, 3.5, 6.1, 0.7, 7.2,
        0.1, 2.1, 3.2, 1.0, 1.7, 2.9, 3.5, 6.1, 0.7, 7.2,
        0.1, 2.1, 3.2, 1.0, 1.7, 2.9, 3.5, 6.1, 0.7, 7.2,
        0.1, 2.1, 3.2, 1.0, 1.7, 2.9, 3.5, 6.1, 0.7, 7.2,
        0.1, 2.1, 3.2, 1.0, 1.7, 2.9, 3.5, 6.1, 0.7, 7.2,
        0.1, 2.1, 3.2, 1.0, 1.7, 2.9, 3.5, 6.1, 0.7, 7.2,
        0.1, 2.1, 3.2, 1.0, 1.7, 2.9, 3.5, 6.1, 0.7, 7.2,
        0.8, 2.8, 1.2, 1.0, 1.7, 2.9, 4.1, 8.1, 3.7, 7.2,
        0.8, 2.8, 1.2, 1.0, 1.7, 2.9, 4.1, 8.1, 3.7, 7.2,
        0.8, 2.8, 1.2, 1.0, 1.7, 2.9, 4.1, 8.1, 3.7, 7.2,
        0.8, 2.8, 1.2, 1.0, 1.7, 2.9, 4.1, 8.1, 3.7, 7.2,
        0.8, 2.8, 1.2, 1.0, 1.7, 2.9, 4.1, 8.1, 3.7, 7.2,
        0.8, 2.8, 1.2, 1.0, 1.7, 2.9, 4.1, 8.1, 3.7, 7.2,
        0.8, 2.8, 1.2, 1.0, 1.7, 2.9, 4.1, 8.1, 3.7, 7.2,
        0.8, 2.8, 1.2, 1.0, 1.7, 2.9, 4.1, 8.1, 3.7, 7.2,
        0.8, 2.8, 1.2, 1.0, 1.7, 2.9, 4.1, 8.1, 3.7, 7.2,
        0.8, 2.8, 1.2, 1.0, 1.7, 2.9, 4.1, 8.1, 3.7, 7.2,
        ],
        &vec![1, 200]
    );
    let x = Tensor::<f32>::new(
        vec![
        0.8, 2.8, 1.2, 1.0, 1.7, 2.9, 4.1, 8.1, 3.7, 7.2,
        0.8, 2.8, 1.2, 1.0, 1.7, 2.9, 4.1, 8.1, 3.7, 7.2,
        0.8, 2.8, 1.2, 1.0, 1.7, 2.9, 4.1, 8.1, 3.7, 7.2,
        0.8, 2.8, 1.2, 1.0, 1.7, 2.9, 4.1, 8.1, 3.7, 7.2,
        0.8, 2.8, 1.2, 1.0, 1.7, 2.9, 4.1, 8.1, 3.7, 7.2,
        0.8, 2.8, 1.2, 1.0, 1.7, 2.9, 4.1, 8.1, 3.7, 7.2,
        0.8, 2.8, 1.2, 1.0, 1.7, 2.9, 4.1, 8.1, 3.7, 7.2,
        0.8, 2.8, 1.2, 1.0, 1.7, 2.9, 4.1, 8.1, 3.7, 7.2,
        0.8, 2.8, 1.2, 1.0, 1.7, 2.9, 4.1, 8.1, 3.7, 7.2,
        0.8, 2.8, 1.2, 1.0, 1.7, 2.9, 4.1, 8.1, 3.7, 7.2,
        0.1, 2.1, 3.2, 1.0, 1.7, 2.9, 3.5, 6.1, 0.7, 7.2,
        0.1, 2.1, 3.2, 1.0, 1.7, 2.9, 3.5, 6.1, 0.7, 7.2,
        0.1, 2.1, 3.2, 1.0, 1.7, 2.9, 3.5, 6.1, 0.7, 7.2,
        0.1, 2.1, 3.2, 1.0, 1.7, 2.9, 3.5, 6.1, 0.7, 7.2,
        0.1, 2.1, 3.2, 1.0, 1.7, 2.9, 3.5, 6.1, 0.7, 7.2,
        0.1, 2.1, 3.2, 1.0, 1.7, 2.9, 3.5, 6.1, 0.7, 7.2,
        0.1, 2.1, 3.2, 1.0, 1.7, 2.9, 3.5, 6.1, 0.7, 7.2,
        0.1, 2.1, 3.2, 1.0, 1.7, 2.9, 3.5, 6.1, 0.7, 7.2,
        0.1, 2.1, 3.2, 1.0, 1.7, 2.9, 3.5, 6.1, 0.7, 7.2,
        0.1, 2.1, 3.2, 1.0, 1.7, 2.9, 3.5, 6.1, 0.7, 7.2,
        ],
        &vec![1, 200]
    );
    g.bench_function("Rayon", 
        |b| b.iter(|| swiglu(&mut y, &x))
    );
    g.bench_function("Common", 
        |b| b.iter(|| uswiglu(&mut y, &x))
    );
    g.finish();
}

criterion_group!(
    benches,
    bench_rms_norm,
    bench_matmul,
    bench_swiglu
);
criterion_main!(benches);
