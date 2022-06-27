use std::{fs, io, fmt::{Formatter, Display, Result}};
use std::collections::HashMap;
use std::io::Write;


#[derive(Debug, Clone)]
pub struct Product {
    time: f32,
    amount: i8,
}

impl Product {
    fn new(time: f32, amount: i8) -> Product {
        let product = Product{ time, amount};
        product
    }
}

#[derive(Debug, Clone)]
struct Recipe {
    recipe_length: usize,
    sub_products: Vec<ProductKind>,
    sub_prod_amount: Vec<i8>,
}

#[derive(PartialEq, Eq, Hash, Debug, Clone)]
struct ProductKind {
    name: String,
}

impl ProductKind {
    fn new (name: String) -> ProductKind {
        ProductKind {name}
    }
}

impl Display for ProductKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{}", self.name)
    }
}

#[derive(Debug, Clone)]
pub struct ProductList {
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

impl Display for Node {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "({}: {})", self.product_kind, self.amount)
    }
}


#[derive(Debug)]
struct Tree {
    parent: Node,
    indent: usize,
    children: Vec<Box<Tree>>,
}

impl Display for Tree {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let indentation = "\t".repeat(self.indent as usize);
        write!(f, "{}{}", indentation, self.parent)
    }
}

impl Tree {
    fn new (node: Node, indent: usize, data: &ProductList) -> Tree {
        let mut children = vec![];
        let node_calc = generate_result(node.clone(), data);
        for sub_node in node_calc {
            let boxed_sub_tree = Box::new(Tree::new(sub_node, indent + 1, data));
            children.push(boxed_sub_tree)
        }
        let tree = Tree { parent: node, indent, children };
        return tree
    }
    fn traverse (&self) {
        println!("{}", self);
        for node in &self.children {
            node.traverse()
        }
    }
}

fn generate_result(node: Node, data: &ProductList) -> Vec<Node> {
    let mut result_vec = vec![];
    // scalar references items per second
    let sub = &data.map[&node.product_kind.clone()];
    let scalar = node.amount / &sub.0.time;
    for index in 0..sub.1.recipe_length {
        let sub_time = data.get_product(sub.1.sub_products[index].clone());
        match sub_time {
            Some(tuple) =>
                result_vec.push(
                    Node {
                        product_kind: sub.1.sub_products[index].clone(),
                        amount:  scalar * sub.1.sub_prod_amount[index] as f32 * tuple.0.time / tuple.0.amount as f32
                    }
                ),
            None => {}
        }
    }
    result_vec
}

fn parse_file(filename: &str) -> ProductList {
    let mut map: HashMap<ProductKind, (Product, Recipe)> = HashMap::new();

    let file = fs::read_to_string(filename)
        .expect("failed to read file");
    for line in file.split('\n') {
        let parse_line: Vec<&str> = line.split(',').collect();
        let p = Product::new(
            parse_line[1].parse().unwrap(),
            parse_line[2].parse().unwrap()
        );
        let r= parse_recipe(&parse_line[3..]);

        map.insert(ProductKind::new((&*parse_line[0].to_string()).parse().unwrap()), (p, r));

    }
    return ProductList { map };
}

fn parse_recipe(recipe_list: &[&str]) -> Recipe {
    let mut products_vec = vec![];
    let mut amount_vec = vec![];
    for sub in recipe_list {
        let sub_vec = sub.split(':').collect::<Vec<&str>>();
        products_vec.push(ProductKind::new(sub_vec[0].to_string()));
        amount_vec.push(sub_vec[1].parse().unwrap());
    }
    Recipe { recipe_length: products_vec.len(), sub_products: products_vec, sub_prod_amount: amount_vec }
}

fn parse_input(data: &ProductList) -> Option<Node> {
    let mut io_input = String::new();
    print!("> ");
    io::stdout().flush().expect("TODO: panic message");
    io::stdin().read_line(&mut io_input).expect("TODO: panic message");
    io_input = io_input.replace("\n", "").replace(" ", "");
    let io_input_array = io_input.split(':').collect::<Vec<&str>>();
    match io_input_array[..] {
        [name, amount] => {
            let test = amount.parse::<f32>().is_ok();
            if !test {
                println!("Invalid Input!: {} must be a number\ntry again", amount);
                parse_input(data);
            }
            let node = Node {
                product_kind: ProductKind::new(name.to_string()),
                amount: amount.parse().unwrap(),
            };
            if !data.map.contains_key(&node.product_kind) {
                println!("Invalid Input!: {} is not a valid product\ntry again", name);
                parse_input(data);
            }
            Some(node)
        },
        _ => {
            println!("Invalid Input!: try again");
            parse_input(data)
        }
    }
}

fn main() {
    let data = parse_file("products.csv");
    println!("------------------------------------------------------");
    println!("Give name of the product and amount separated by ':'\nexample: 'green_circuit: 10'");
    println!("------------------------------------------------------");
    match parse_input(&data) {
        Some(node) => Tree::new(node, 0, &data).traverse(),
        None => {}
    }
}