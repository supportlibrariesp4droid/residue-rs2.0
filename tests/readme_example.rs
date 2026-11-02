//! The example from README.md, kept here so that it always compiles and runs.

#[test]
#[allow(unused_variables, clippy::identity_op, clippy::erasing_op)]
fn readme_example() {
    use residue::{rns_base, Extender, Rns, RnsMatrix, RnsVec};

    rns_base!(Narrow: largest 4);
    rns_base!(Wide: largest 9);

    let a: Rns<Narrow, 4> = "-123456789012345678901234567890".parse().unwrap();
    let b = Rns::<Narrow, 4>::from_i64(42).unwrap();
    assert!(a < b);
    let c = a * b;

    let ext = Extender::<Narrow, 4, Wide, 9>::new();
    let big = ext.extend(&a).unwrap() * ext.extend(&a).unwrap();

    let xs = RnsVec::<Wide, 9>::from_i128s(&[1, -2, 3]).unwrap();
    let ys = RnsVec::<Wide, 9>::from_i128s(&[4, 5, -6]).unwrap();
    assert_eq!(xs.dot(&ys).unwrap().to_i128(), Some(1 * 4 - 2 * 5 - 3 * 6));
    assert_eq!((&xs * &ys).to_i128s(), Ok(vec![4, -10, -18]));

    let m = RnsMatrix::<Wide, 9>::from_i128s(2, 2, &[3, 1, 4, 1]).unwrap();
    assert_eq!(m.determinant().unwrap().to_i128(), Some(-1));
    let m2 = &m * &m;

    let x = Rns::<Wide, 9>::from_i64(91).unwrap();
    let y = Rns::<Wide, 9>::from_i64(7).unwrap();
    assert_eq!(x.div_exact(y).unwrap().to_i128(), Some(13));
    assert_eq!((x % Rns::from_i64(10).unwrap()).to_i128(), Some(1));
}

/// The modular example from README.md.
#[test]
fn readme_modular_example() {
    use residue::{rns_base, ModVec, MontRns};

    rns_base!(B1: largest 6);
    rns_base!(B2: largest 6 after 6);

    let m = [u64::MAX, u64::MAX >> 1];
    let ctx = MontRns::<B1, 6, B2, 6>::new(&m).unwrap();
    let a = ctx.from_u128(3);
    let b = ctx.from_u128(1 << 100);
    assert_eq!((a * a + a).to_u128(), Some(12));
    assert_eq!((b * b).to_u128(), Some(1 << 73));
    assert_eq!(a.pow(&[u64::MAX - 1, u64::MAX >> 1]), ctx.one());
    assert_eq!(a * a.inv_prime(), ctx.one());

    let v = ModVec::from_elems(&ctx, &[a, b, a + b]);
    let w = v.pow(&[65537]);
    assert_eq!(w.get(2), Some((a + b).pow(&[65537])));
}
