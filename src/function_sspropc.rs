use fftw::plan::*;
use fftw::types::*;
use std::f64::consts::PI;

#[allow(dead_code)]
fn abs2(x: c64) -> f64 {
    x.re * x.re + x.im * x.im
}

#[allow(dead_code)]
fn prodr(x: c64, y: c64) -> f64 {
    x.re * y.re + x.im * y.im
}

#[allow(dead_code)]
fn prodi(x: c64, y: c64) -> f64 {
    x.re * y.im - x.im * y.re
}

#[allow(dead_code)]
fn cmult(a: &mut Vec<c64>, b: &[c64], c: &[c64]) {
    for i in 0..b.len() {
        a[i].re = b[i].re * c[i].re - b[i].im * c[i].im;
        a[i].im = b[i].re * c[i].im + b[i].im * c[i].re;
    }
}

#[allow(dead_code)]
fn cscale(a: &mut Vec<c64>, b: &[c64], factor: f64) {
    for i in 0..b.len() {
        a[i].re = factor * b[i].re;
        a[i].im = factor * b[i].im;
    }
}

#[allow(dead_code)]
fn ssconverged(a: &[c64], b: &[c64], t: f64) -> bool {
    let mut num: f64 = 0.0;
    let mut denom: f64 = 0.0;
    let nt = b.len() as f64;

    for i in 0..b.len() {
        denom += b[i].re * b[i].re + b[i].im * b[i].im;
        num += (b[i].re - a[i].re / nt) * (b[i].re - a[i].re / nt)
            + (b[i].im - a[i].im / nt) * (b[i].im - a[i].im / nt);
    }

    num / denom < t
}

#[allow(dead_code)]
pub fn sspropc(
    u0: &[c64],
    dt: f64,
    dz: f64,
    n1: u64,
    alp: &[f64],
    betap: &[f64],
    gamma: f64,
    tr: f64,
    s: f64,
    maxiter: u8,
    tol: f64,
) -> Vec<c64> {
    let n = u0.len();
    let mut u0_temp = vec![c64::new(0.0, 0.0); n];
    let mut u1 = vec![c64::new(0.0, 0.0); n];
    let mut ufft = vec![c64::new(0.0, 0.0); n];
    let mut uhalf = vec![c64::new(0.0, 0.0); n];
    let mut uhalf_out = vec![c64::new(0.0, 0.0); n];
    let mut uv = vec![c64::new(0.0, 0.0); n];
    let mut uv_out = vec![c64::new(0.0, 0.0); n];
    let mut u2 = vec![c64::new(0.0, 0.0); n];
    let mut halfstep = vec![c64::new(0.0, 0.0); n];

    let mut p1: C2CPlan64 = C2CPlan::aligned(&[n], Sign::Forward, Flag::PATIENT).unwrap();
    let mut p2: C2CPlan64 = C2CPlan::aligned(&[n], Sign::Forward, Flag::PATIENT).unwrap();
    let mut ip1: C2CPlan64 = C2CPlan::aligned(&[n], Sign::Backward, Flag::PATIENT).unwrap();
    let mut ip2: C2CPlan64 = C2CPlan::aligned(&[n], Sign::Backward, Flag::PATIENT).unwrap();

    let nz = n1;
    let nalp = alp.len();
    let nbeta = betap.len();
    
    let mut phase: f64;
    let mut alpha: f64;
    
    let mut ua: &c64;
    let mut ub: &c64;
    let mut uc: &c64;
    let mut nlp_re: f64;
    let mut nlp_im: f64;

    /* compute vector of angular frequency components */
    /* MATLAB equivalent:  w = wspace(tv); */
    let mut w = vec![0.0_f64; n];
    for (i, w_in) in w.iter_mut().enumerate().take((n - 1) / 2) {
        *w_in = 2. * PI * (i as f64) / (dt * (n as f64));
    }
    for (i, w_in) in w.iter_mut().enumerate().take(n).skip((n - 1) / 2) {
        *w_in = 2. * PI * (i as f64) / (dt * (n as f64)) - 2. * PI / dt;
    }

    /* compute halfstep and initialize u0 and u1 */
    for i in 0..n {
        if nbeta != n {
            let mut fii = 1.;
            let mut wii = 1.;
            phase = 0.0;
            for (j, beta) in betap.iter().enumerate() {
                phase += wii * (beta / fii);
                fii *= (j as f64) + 1.;
                wii *= w[i];
            }
        } else {
            phase = betap[i];
        }
        if nalp == n {
            alpha = alp[i];
        } else {
            alpha = alp[0];
        }

        halfstep[i].re = (-alpha * dz / 4.).exp() * (phase * dz / 2.).cos();
        halfstep[i].im = -((alpha * dz / 4.).exp()) * (phase * dz / 2.).sin();

        u0_temp[i] = u0[i];
        u1[i] = u0[i];
        u2[i] = u0[i];
    }

    println!("Performing split-step iterations...");

    
    p1.c2c(&mut u0_temp, &mut ufft).unwrap(); /* ufft = fft(u0) */

    let nt = n as f64;
    for _ in 0..nz {
        cmult(&mut uhalf, &halfstep, &ufft); /* uhalf = halfstep.*ufft */
        ip1.c2c(&mut uhalf, &mut uhalf_out).unwrap(); /* uhalf = nt*ifft(uhalf) */
        uhalf = uhalf_out.clone();
        let mut count = 0;
        for _ in 0..maxiter {
            if (tr == 0.0) && (s == 0.0) {
                for j in 0..n {
                    phase = gamma
                        * (u1[j].re * u1[j].re
                            + u1[j].im * u1[j].im
                            + u2[j].re * u2[j].re
                            + u2[j].im * u2[j].im)
                        * dz
                        / 2.;
                    uv[j].re = (uhalf[j].re * phase.cos() + uhalf[j].im * phase.sin()) / nt;
                    uv[j].im = (-uhalf[j].re * phase.sin() + uhalf[j].im * phase.cos()) / nt;
                }
            } else {
                let mut j = 0;
                ua = &u1[n - 1];
                ub = &u1[j];
                uc = &u1[j + 1];
                nlp_im =
                    -s * (abs2(*uc) - abs2(*ua) + prodr(*ub, *uc) - prodr(*ub, *ua)) / (4. * PI * dt);
                nlp_re = abs2(*ub) - tr * (abs2(*uc) - abs2(*ua)) / (2. * dt)
                    + s * (prodi(*ub, *uc) - prodi(*ub, *ua)) / (4. * PI * dt);

                ua = &u2[n - 1];
                ub = &u2[j];
                uc = &u2[j + 1];
                nlp_im +=
                    -s * (abs2(*uc) - abs2(*ua) + prodr(*ub, *uc) - prodr(*ub, *ua)) / (4. * PI * dt);
                nlp_re += abs2(*ub) - tr * (abs2(*uc) - abs2(*ua)) / (2. * dt)
                    + s * (prodi(*ub, *uc) - prodi(*ub, *ua)) / (4. * PI * dt);

                nlp_re *= gamma * dz / 2.;
                nlp_im *= gamma * dz / 2.;

                uv[j].re = (uhalf[j].re * nlp_re.cos() * nlp_im.exp()
                    + uhalf[j].im * nlp_re.sin() * nlp_im.exp())
                    / nt;
                uv[j].im = (-uhalf[j].re * nlp_re.sin() * nlp_im.exp()
                    + uhalf[j].im * nlp_re.cos() * nlp_im.exp())
                    / nt;

                for j in 1..(n - 1) {
                    ua = &u1[j - 1];
                    ub = &u1[j];
                    uc = &u1[j + 1];
                    nlp_im =
                        -s * (abs2(*uc) - abs2(*ua) + prodr(*ub, *uc) - prodr(*ub, *ua)) / (4. * PI * dt);
                    nlp_re = abs2(*ub) - tr * (abs2(*uc) - abs2(*ua)) / (2. * dt)
                        + s * (prodi(*ub, *uc) - prodi(*ub, *ua)) / (4. * PI * dt);

                    ua = &u2[j - 1];
                    ub = &u2[j];
                    uc = &u2[j + 1];
                    nlp_im +=
                        -s * (abs2(*uc) - abs2(*ua) + prodr(*ub, *uc) - prodr(*ub, *ua)) / (4. * PI * dt);
                    nlp_re += abs2(*ub) - tr * (abs2(*uc) - abs2(*ua)) / (2. * dt)
                        + s * (prodi(*ub, *uc) - prodi(*ub, *ua)) / (4. * PI * dt);

                    nlp_re *= gamma * dz / 2.;
                    nlp_im *= gamma * dz / 2.;

                    uv[j].re = (uhalf[j].re * nlp_re.cos() * nlp_im.exp()
                        + uhalf[j].im * nlp_re.sin() * nlp_im.exp())
                        / nt;
                    uv[j].im = (-uhalf[j].re * nlp_re.sin() * nlp_im.exp()
                        + uhalf[j].im * nlp_re.cos() * nlp_im.exp())
                        / nt;
                }

                j = n;
                ua = &u1[j - 1];
                ub = &u1[j];
                uc = &u1[0];
                nlp_im =
                    -s * (abs2(*uc) - abs2(*ua) + prodr(*ub, *uc) - prodr(*ub, *ua)) / (4. * PI * dt);
                nlp_re = abs2(*ub) - tr * (abs2(*uc) - abs2(*ua)) / (2. * dt)
                    + s * (prodi(*ub, *uc) - prodi(*ub, *ua)) / (4. * PI * dt);

                ua = &u2[j - 1];
                ub = &u2[j];
                uc = &u2[0];
                nlp_im +=
                    -s * (abs2(*uc) - abs2(*ua) + prodr(*ub, *uc) - prodr(*ub, *ua)) / (4. * PI * dt);
                nlp_re += abs2(*ub) - tr * (abs2(*uc) - abs2(*ua)) / (2. * dt)
                    + s * (prodi(*ub, *uc) - prodi(*ub, *ua)) / (4. * PI * dt);

                nlp_re *= gamma * dz / 2.;
                nlp_im *= gamma * dz / 2.;

                uv[j].re = (uhalf[j].re * nlp_re.cos() * nlp_im.exp()
                    + uhalf[j].im * nlp_re.sin() * nlp_im.exp())
                    / nt;
                uv[j].im = (-uhalf[j].re * nlp_re.sin() * nlp_im.exp()
                    + uhalf[j].im * nlp_re.cos() * nlp_im.exp())
                    / nt;
            }
            p2.c2c(&mut uv, &mut uv_out).unwrap();
            uv = uv_out.clone();
            cmult(&mut ufft, &uv, &halfstep);
            ip2.c2c(&mut ufft, &mut uv).unwrap();
            if ssconverged(&uv, &u1, tol) {
                cscale(&mut u1, &uv, 1.0 / nt);
                break;
            } else {
                cscale(&mut u1, &uv, 1.0 / nt);
            }
            count += 1;
        }
        if count == maxiter {
            println!("Failed to converge.");
        }
        u2 = u1.clone();
    }
    println!("done.");
    u2
}
