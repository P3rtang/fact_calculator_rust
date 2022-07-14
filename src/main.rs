use std::{fs, fmt::{Formatter, Display, Result}, io};
use std::borrow::Borrow;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::io::Write;
use std::iter::{Enumerate, Peekable};
use std::process::exit;
use std::str::Chars;
use ansi_term::Colour;
use crate::Error::{KeyError, SyntaxError};
use crate::TokenKind::Colon;

static QUIT: bool = false;

#[derive(Debug)]
enum CommandKind {
    Help,
    Calc,
    List(ProductList),
    New
}

struct Command {
    kind: CommandKind,
    args: Option<Vec<Token>>
}

impl Command {
    fn run(&self) {
        match &self.kind {
            CommandKind::Help    => println!("{}", self),
            CommandKind::Calc    => todo!("Calc not yet implemented"),
            CommandKind::List(_) => print!("{}", self),
            CommandKind::New     => {},
        }
    }
}

impl Display for Command {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match &self.kind {
            CommandKind::Help => { write!(f, "{}", fs::read_to_string("help.txt").expect("failed to read file")) },
            CommandKind::List(data) => { write!(f, "{}", data) },
            err => { panic!("Tried to print non-printable Command {:?}", err) }
        }
    }
}


#[derive(Debug)]
enum Error {
    SyntaxError (String, usize, String),
    KeyError    (String, usize, String),
}

impl Error {
    fn raise(&self) {
        self.show();
        exit(1);
    }
    fn show(&self) {
        match self {
            SyntaxError(_, _, _) => { print!("{}", self) }
            KeyError   (_, _, _) => { print!("{}", self) }
        }
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            SyntaxError(_, loc, info) => {
                writeln!(f, "{}^ SyntaxError: {}", " ".repeat(*loc + 2), info)
            }
            KeyError(_, loc, info) => {
                writeln!(f, "{}^ KeyError: {}", " ".repeat(*loc + 2), info)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Product {
    kind: ProductKind,
    time: f32,
    amount: i8,
    recipe_products: HashMap<ProductKind, i8>,
}

impl PartialEq<Product> for Product {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
    }
}
impl PartialEq<ProductKind> for Product {
    fn eq(&self, other: &ProductKind) -> bool {
        self.kind == *other
    }
}

impl Eq for Product {}

impl PartialOrd<Self> for Product {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(&other))
    }
}

impl Ord for Product {
    fn cmp(&self, other: &Self) -> Ordering {
        self.kind.cmp(&other.kind)
    }
}

impl Product {
    fn new(kind: ProductKind, time: f32, amount: i8, recipe_products: HashMap<ProductKind, i8>) -> Product {
        let product = Product{ kind, time, amount, recipe_products};
        product
    }
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

impl PartialOrd<Self> for ProductKind {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(&other))
    }
}

impl Ord for ProductKind {
    fn cmp(&self, other: &Self) -> Ordering {
        self.name.cmp(&other.name)
    }
}

#[derive(Debug, Clone)]
pub struct ProductList {
    list: Vec<Product>
}

impl Display for ProductList {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let mut display = "".to_string();
        let mut key_slice = self.list.clone();
        key_slice.sort();
        for key in key_slice {
            display += &format!("{}\n", key.kind)
        }
        writeln!(f, "{}", Colour::RGB(77, 208, 225).paint(display))
    }
}

impl ProductList {
    fn get_product(&self, product_kind: &ProductKind) -> Option<&Product> {
        for p in &self.list {
            if p == product_kind {
                return Some(p)
            }
        }
        return None
    }
    fn contains(&self, product_kind: &ProductKind) -> bool {
        return self.get_product(product_kind) != None
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

#[derive(Debug, Clone)]
enum TokenKind {
    Comment,
    Expr,
    Colon,
    Comma,
    Dash,
}

#[derive(Debug, Clone)]
struct Token {
    kind: TokenKind,
    text: String,
    loc: usize,
}

impl Token {
    fn new(kind: TokenKind, text:String, loc: usize) -> Self { Token {kind, text, loc} }
}

#[derive(Debug, Clone)]
struct Lexer<Char: Iterator<Item=char>> {
    input: String,
    chars: Peekable<Enumerate<Char>>
}

impl <Char: Iterator<Item=char>> Iterator for Lexer<Char> {
    type Item = Token;

    fn next(&mut self) -> Option<Self::Item> {
        use crate::TokenKind::{Colon, Comma, Comment, Dash, Expr};
        let mut text = String::new();
        if let Some((loc, x)) = self.chars.next() {
            text.push(x);
            match x {
                ':' => Some(Token::new(Colon, text, loc)),
                ',' => Some(Token::new(Comma, text, loc)),
                '#' => Some(Token::new(Comment, text, loc)),
                '-' => Some(Token::new(Dash, text, loc)),
                ' ' => self.next(),
                '\n' => None,
                char if char.is_alphanumeric() => {
                    while self.chars.peek().unwrap().1.is_alphanumeric() {
                        text.push(self.chars.next().unwrap().1)
                    };
                    return Some(Token::new(Expr, text, loc))
                }
                _ => { SyntaxError(self.input.clone(), loc, "Unexpected Character".to_string()).show(); None}
            }
        } else { None }
    }
}

impl <Char: Iterator<Item=char>> Lexer<Char> {
    fn new(input: String, chars: Char) -> Self{
        Self { input, chars: chars.enumerate().peekable() }
    }
}

fn generate_result(node: Node, data: &ProductList) -> Vec<Node> {
    let mut result_vec = vec![];
    // scalar references items per second
    let sub = data.get_product(&node.product_kind);
    match sub {
        None => {},
        Some(product) => {
            let scalar = node.amount / product.time;
            for (product_kind, amount) in product.recipe_products.clone() {
                let sub_time = data.get_product(&product_kind);
                match sub_time {
                    Some(product) =>
                        result_vec.push(
                            Node {
                                product_kind,
                                amount:  scalar * amount as f32 * product.time / product.amount as f32
                            }
                        ),
                    None => {}
                }
            }}
    }

    result_vec
}

fn parse_file(filename: &str) -> ProductList{

    fn parse_recipe(recipe_list: &[&str]) -> HashMap<ProductKind, i8> {
        let mut return_map = HashMap::new();
        for sub in recipe_list {
            let sub_vec = sub.split(':').collect::<Vec<&str>>();
            return_map.insert(ProductKind::new(sub_vec[1].to_string()), sub_vec[1].parse().unwrap());
        }
        return return_map
    }

    let mut list = vec![];

    let file = fs::read_to_string(filename)
        .expect("failed to read file");
    for line in file.split('\n') {
        match line.chars().peekable().peek() {
            Some('#') => { },
             _  => {
                let parse_line: Vec<&str> = line.split(',').collect();
                let p = Product::new(
                    ProductKind::new(parse_line[0].to_string()),
                    parse_line[1].parse().unwrap(),
                    parse_line[2].parse().unwrap(),
                    parse_recipe(&parse_line[3..])
                );
                list.push(p);
            }
        }
    }
    return ProductList { list };
}

fn parse_lexer(mut lexer: &mut Lexer<Chars<'_>>) -> Option<Command> {
    if let Some(token) = lexer.next() {
        match token.kind {
            TokenKind::Expr => {
                match token.text.to_ascii_lowercase().as_str() {
                    "exit" => exit(0),
                    "new"  => {
                        let args:Vec<Token> = lexer.collect();
                        if args.len() > 1{
                            let err = args[1].clone();
                            SyntaxError(err.text, err.loc, format!("Expected 1 arguments but got {}", args.len())).show();
                            return None
                        }
                        else if args.len() == 0 {
                            SyntaxError("".to_string(), 4, format!("Expected 1 arguments but got {}", args.len())).show();
                            return None
                        }
                        Some(Command {kind:CommandKind::New, args: Some(args) })
                    },
                    "help" => { None }
                    err => {
                        KeyError(err.to_string(), token.loc, "Unexpected Command use --help for possible commands".to_string()).show();
                        None
                    }
                }
            },
            _ => todo!(),
        }
    } else { None }
}

fn main() {
    let _data = parse_file("products.csv");
    println!("------------------------------------------------------");
    println!("Give name of the product and amount separated by ':'\nexample: 'green_circuit: 10'");
    println!("------------------------------------------------------");
    // match parse_input(&data) {
    //     Some(node) => Tree::new(node, 0, &data).traverse(),
    //     None => {}
    // }
    while !QUIT {
        let mut io_input = String::new();
        print!("> ");
        io::stdout().flush().expect("ERROR: Failed to print io::stdout buffer");
        io::stdin().read_line(&mut io_input).expect("ERROR: Failed to read io::stdin");
        let mut lexer = Lexer::new(io_input.clone(), io_input.chars());
        if let Some(command) = parse_lexer(&mut lexer) { command.run() } else {};
    }
}