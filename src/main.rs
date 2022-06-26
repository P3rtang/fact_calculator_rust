use std::fs;
use std::collections::HashMap;
use std::time;
use std::fmt;
use std::fmt::Formatter;

#[derive(Debug)]
pub struct Product {
    kind: ProductKind,
    time: f32,
    amount: i8,
}

impl Product {
    fn new(name: String, time: f32, amount: i8) -> Product {
        let product = Product{ kind: ProductKind::Name(name), time, amount};
        product
    }
}

#[derive(Debug)]
struct Recipe {
    recipe_length: usize,
    sub_products: Vec<ProductKind>,
    sub_prod_amount: Vec<i8>,
}

#[derive(PartialEq, Eq, Hash, Debug, Clone)]
enum ProductKind {
    Name(String),
}

impl fmt::Display for ProductKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

#[derive(Debug)]
struct ProductList {
    map: HashMap<ProductKind, (Product, Recipe)>
}

impl ProductList {
    fn get_product(&self, product_kind: ProductKind) -> Option<&(Product, Recipe)> {
        for (p, data) in &self.map {
            if p == &product_kind {
                return Some(&data)
            }
        }
        return None
    }
}

#[derive(Debug, Clone)]
struct Node {
    product_kind: ProductKind,
    amount: f32,
}

impl fmt::Display for Node {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "({}: {})", self.product_kind, self.amount)
    }
}

struct Tree {
    parent: Node,
    children: Vec<Box<Tree>>,
}

impl Tree {
    fn print_indent (&self) {
        
    }
}

#[derive(Debug)]
struct Calculator {
    input: Node,
    data: ProductList,
}

impl Calculator {
    // sub.amount * sub.time / sub.amount / scalar
    fn generate_result(&mut self) -> Vec<Node> {
        let mut result_vec = vec![self.input.clone()];
        // scalar references items per second
        let sub = &self.data.map[&self.input.product_kind];
        let scalar = self.input.amount / &sub.0.time;
        for index in 0..sub.1.recipe_length {
            let sub_time = self.data.get_product(sub.1.sub_products[index].clone());
            match sub_time {
                Some(tuple) =>
                    result_vec.push(
                        Node {
                            product_kind: sub.1.sub_products[index].clone(),
                            amount:  scalar * sub.1.sub_prod_amount[index] as f32 * tuple.0.time
                        }
                    ),
                None => {}
            }
        }
        result_vec
    }
}

fn parse_file(filename: &str) -> ProductList {
    let mut map: HashMap<ProductKind, (Product, Recipe)> = HashMap::new();

    let file = fs::read_to_string(filename)
        .expect("failed to read file");
    for line in file.split('\n') {
        let parse_line: Vec<&str> = line.split(',').collect();
        let p = Product::new(
            parse_line[0].to_string(),
            parse_line[1].parse().unwrap(),
            parse_line[2].parse().unwrap()
        );
        let r= parse_recipe(&parse_line[3..]);

        map.insert(ProductKind::Name((&*parse_line[0].to_string()).parse().unwrap()), (p, r));

    }
    return ProductList { map };
}

fn parse_recipe(recipe_list: &[&str]) -> Recipe {
    let mut products_vec = vec![];
    let mut amount_vec = vec![];
    for sub in recipe_list {
        let sub_vec = sub.split(':').collect::<Vec<&str>>();
        products_vec.push(ProductKind::Name(sub_vec[0].to_string()));
        amount_vec.push(sub_vec[1].parse().unwrap());
    }
    Recipe { recipe_length: products_vec.len(), sub_products: products_vec, sub_prod_amount: amount_vec }
}

fn main() {
    let data = parse_file("products.csv");
    let mut calc = Calculator {input: Node {product_kind: ProductKind::Name("green_circuit".to_string()), amount: 10.0 }, data };
    let result = calc.generate_result();
    for node in result {
        println!("{}", node)
    }
}