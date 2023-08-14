pub fn to(x: i64, y: i64, credibility: Option<u8>) -> Dist {
    if x > 0 {
        norm(x, y, credibility)
    }
}
