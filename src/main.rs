mod function_sspropc;
mod function_tools;

use csv::{Reader, Writer};
use fftw::types::*;

use function_sspropc::sspropc;

fn main() {
    // 读取数据
    let n = (2_u32).pow(15);
    let mut date = vec![c64::new(0.0, 0.0); n as usize];
    let mut rdr = Reader::from_path("date.csv").unwrap();
    for (i, result) in rdr.records().take(4096).enumerate() {
        let record = result.unwrap();
        date[i] = c64::new(record[0].parse::<f64>().unwrap(), 0.);
    }

    // 采样时间
    let dt: f64 = (0.1_f64).powf(3.0);

    // 光纤长度，步径，绘图数
    let z: f64 = 3. * (0.1_f64).powf(2.0);
    let nz: f64 = 6000.;
    let nplot: f64 = 20.;
    let n1: f64 = (nz / nplot + 0.4).round();
    let nz: f64 = n1 * nplot;
    let dz: f64 = z / nz;

    // 光纤参数
    let beta2: f64 = -1.31;
    let beta3: f64 = -4. * (0.1_f64).powf(2.0);
    let beta4: f64 = -1.14 * (0.1_f64).powf(4.0);
    let beta5: f64 = 1.73 * (0.1_f64).powf(7.0);
    let beta6: f64 = -9.8 * (0.1_f64).powf(10.0);
    let betap: Vec<f64> = vec![0., 0., beta2, beta3, beta4, beta5, beta6];
    let gamma: f64 = 0.028;
    let alp: Vec<f64> = vec![5.];

    let tr: f64 = 0.;
    let s: f64 = 0.;

    let maxiter: u8 = 4;
    let tol: f64 = (0.1_f64).powf(5.0);

    let n_iter = nplot as usize;
    let mut u: Vec<Vec<c64>> = vec![vec![c64::new(0., 0.); 4096]; n_iter+1];
    u[0] = date;

    // 迭代计算 得时序
    for i in 0..n_iter {
        println!("i = {} start.", i + 1);
        u[i + 1] = sspropc(
            &u[i], dt, dz, n1 as u64, &alp, &betap, gamma, tr, s, maxiter, tol,
        );
    }

    // 时序数据转换，后导出
    let u_string: Vec<Vec<String>> = u
        .iter()
        .map(|x| x.iter().map(|y| 
                                    if y.im > 0. { 
                                        y.re.to_string() + "+" + &y.im.to_string() + "j" 
                                    } else {
                                        y.re.to_string() + &y.im.to_string() + "j"
                                    })
                        .collect())
        .collect();

    println!("csv exportting!");
    let mut wtr = Writer::from_path("y.csv").unwrap();
    for i in 0..4096 {
        let mut record: Vec<String> = vec!["".to_string(); n_iter+1];
        for (index, j) in u_string.iter().enumerate() {
            record[index] = j[i].clone();
        }
        wtr.write_record(&record).unwrap();
    }
    println!("csv exported!")
}
