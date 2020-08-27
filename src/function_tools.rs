use std::f64::consts::PI;

#[allow(dead_code)]
pub fn wspace(dt: f64, n: usize) -> Vec<f64> {
    let mut w = vec![0.0_f64; n];
    for (i, w_in) in w.iter_mut().enumerate().take((n - 1) / 2) {
        *w_in = 2. * PI * (i as f64) / (dt * (n as f64));
    }
    for (i, w_in) in w.iter_mut().enumerate().take(n).skip((n - 1) / 2) {
        *w_in = 2. * PI * (i as f64) / (dt * (n as f64)) - 2. * PI / dt;
    }
    w
}
