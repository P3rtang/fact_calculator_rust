#![allow(non_snake_case)]
use std::fs;

struct Recipe {
}


struct Product {
    name: String,
    prod_time: f32,
    amount_crafted: i32,
    recipe: Recipe
}

fn get_product_from_string(filename: &str) {
    let contents = fs::read_to_string(filename)
        .expect("Something went wrong reading the file");
    for (row, text) in contents.split('\n').enumerate() {
        println!("line{}: {}", row, text)
    }
}


fn main() {
    get_product_from_string("products.csv")
}