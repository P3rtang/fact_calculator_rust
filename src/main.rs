use std::{fs, fmt::{Formatter, Display, Result}, str::FromStr, io::{stdout, stdin, Write}, cmp::{max, Ordering}, iter::Peekable, process::exit, str::Chars, collections::HashMap};
use termion::{color, event::{Event, Key}, input::{TermRead}, raw::IntoRawMode, cursor, clear, style};
use termion::cursor::{DetectCursorPos, Goto};
use crate::Error::{InputError, IOError, KeyError, ParsingError, SyntaxError, ValueError};

static QUIT: bool = false;

#[derive(Debug)]
enum CommandKind {
    Help,
    Calc,
    List,
    New,
    Edit,
    // TODO: add set command (for basic settings like smelter speed)
}

struct Command {
    kind: CommandKind,
    args: Vec<Token>,
}

impl Command {
    fn run(&self, data: &mut ProductList, settings: &Settings) {
        match &self.kind {
            CommandKind::Help    => println!("{}", self),
            CommandKind::Calc    => {
                match &self.args[..] {
                    [token1, token2] => {
                        let node = Node { product_kind: ProductKind { name: token1.clone().text }, amount: token2.text.parse().unwrap() };
                        let tree = Tree::new(node, 0, data, settings);
                        tree.traverse()
                    }
                    [err,..] if err.kind != TokenKind::NewLine => {
                        SyntaxError(err.clone().text, err.loc, format!("expected 2 arguments for calc found {}: {}", self.args.len(), err.text)).show()
                    }
                    [_newline] => {
                        // TODO: add an error for unknown products
                        let mut interactive_terminal = InteractiveProductTerm::new(format!("product >> "), data);
                        interactive_terminal.start_session();
                        let product = interactive_terminal.input;
                        let mut amount = String::new();

                        print!("\namount  >> ");
                        stdout().flush().expect("IOError could not flush stdout");
                        stdin().read_line(&mut amount).expect("IOError could not read stdin");
                        let parse_amount = match amount.clone().drain(..(amount.len() - 1)).as_str().parse::<f32>() {
                            Ok(out) => out,
                            Err(_) => { ValueError(amount, 9, format!("Could not interpret this as a number defaulting to 1.0")).show(); 1.0 }
                        };
                        let node = Node { product_kind: ProductKind { name: product }, amount: parse_amount };
                        let tree = Tree::new(node, 0, data ,settings);
                        tree.traverse()
                    }
                    [..] => {}
                }
            },
            CommandKind::List    => print!("{}", data),
            CommandKind::New     => {
                match &self.args[..] {
                    [token, amount, time,..] => {
                        let mut recipe = Vec::new();
                        let info = (amount.text.parse::<i8>().unwrap(), time.text.parse::<f32>().unwrap());

                        while let Some(input) = new_product_recipe_dialog() {
                            let rec_part = RecipePart { kind: ProductKind { name: input.0.clone() }, amount: input.1.clone() };
                            if data.contains(&rec_part.kind) {
                                recipe.push(rec_part)
                            } else {
                                ValueError(input.0.clone(), input.1 as usize + 8, format!("The given subproduct {} is not defined", input.0)).show();
                            }
                        }
                        println!("\nAdding product:\n \t{} [production amount:: {}, production time:: {}]", token.text, info.0, info.1);
                        for part in recipe {
                            println!("Adding recipe:\n{}", part)
                        }
                    }
                    [token, ..] => {
                        SyntaxError(token.clone().text, token.loc, format!("Incorrect use of the `new` command")).show()
                    }

                    [] => {
                        // TODO: add a default for product and use it here
                        let new_template = Product {
                            kind: ProductKind { name: "new_name".to_string() },
                            time: 1.0,
                            amount: 1,
                            machine: MachineKind::Factory,
                            recipe_products: vec![]
                        };
                        let mut interactive_session = InteractiveEditTerm::new(new_template, 0);
                        interactive_session.start_session(data);

                        let new_product = interactive_session.get_product();
                        if !interactive_session.escape_pressed {
                            data.add(new_product.clone());
                            save_product_to_file(new_product, "products.csv".to_string())
                        }
                    }
                }
            }
            CommandKind::Edit => {
                // TODO: save edited products to the csv
                println!("Select a product to edit");

                let mut interactive_product_session = InteractiveProductTerm::new(format!("product >> "), data);
                interactive_product_session.start_session();
                if let Some(product) = interactive_product_session.input_get_product() {
                    let mut interactive_edit_session = InteractiveEditTerm::new(product.clone(), 1);
                    interactive_edit_session.start_session(data);
                    if !interactive_edit_session.escape_pressed {
                        let edited_product = interactive_edit_session.get_product();
                        data.remove(product);
                        data.add(edited_product);
                    }
                } else {
                    println!();
                    InputError("".to_string(), 9, format!("Could not find {}", interactive_product_session.input)).show();
                }
            }
        }
    }
}

impl Display for Command {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match &self.kind {
            CommandKind::Help => { write!(f, "{}", fs::read_to_string("help.txt").expect("failed to read file")) },
            err => { panic!("Tried to print non-printable Command {:?}", err) }
        }
    }
}

// TODO: add a location struct for file, x, y position for the exact place of the error
#[derive(Debug)]
enum Error {
    SyntaxError (String, usize, String),
    KeyError    (String, usize, String),
    ValueError  (String, usize, String),
    IOError     (String, usize, String),
    InputError  (String, usize, String),
    ParsingError(String, usize, String),
}

impl Error {
    fn raise(&self) {
        self.show();
        exit(1);
    }
    fn show(&self) {
        match self {
            SyntaxError  (_, _, _) => { eprint!("{}", self) }
            KeyError     (_, _, _) => { eprint!("{}", self) }
            ValueError   (_, _, _) => { eprint!("{}", self) }
            IOError      (_, _, _) => { eprint!("{}", self) }
            InputError   (_, _, _) => { eprint!("{}", self) }
            ParsingError (_, _, _) => { eprint!("{}", self) }
        }
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            SyntaxError(_, loc, info) => {
                writeln!(f, "{}^ SyntaxError: {}", " ".repeat(*loc), info)
            }
            KeyError(_, loc, info)    => {
                writeln!(f, "{}^ KeyError: {}",    " ".repeat(*loc), info)
            }
            ValueError(_, loc, info)  => {
                writeln!(f, "{}^ ValueError: {}",  " ".repeat(*loc), info)
            }
            IOError(_, loc, info)  => {
                writeln!(f, "{}^ IOError: {}",     " ".repeat(*loc), info)
            }
            InputError(_, loc, info)  => {
                writeln!(f, "{}^ InputError: {}",  " ".repeat(*loc), info)
            }
            ParsingError(_, loc, info)  => {
                writeln!(f, "{}^ ParsingError: {}",  " ".repeat(*loc), info)
            }
        }
    }
}

#[derive(Clone, Debug)]
struct RecipePart {
    kind: ProductKind,
    amount: i8,
}

impl Display for RecipePart {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        writeln!(f, "{}:{}", self.kind, self.amount)
    }
}

impl RecipePart {
    fn parse_recipe(tokens: &[Token]) -> Vec<Self> {
        let mut recipe_products = vec![];
        for (index, token) in tokens.into_iter().step_by(2).enumerate() {
            recipe_products.push(RecipePart{ kind: ProductKind { name: token.text.clone() }, amount: tokens[1 + index*2].text.parse().unwrap() })
        }
        return recipe_products
    }
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
enum MachineKind {
    Smelter,
    Factory,
    Chemical,
    Refinery,
    Miner,
}

impl Default for MachineKind {
    fn default() -> Self {
        MachineKind::Factory
    }
}

impl FromStr for MachineKind {
    type Err = ();

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        return match s {
            "smelter" => Ok(Self::Smelter),
            "factory" => Ok(Self::Factory),
            "chemical" => Ok(Self::Chemical),
            "refinery" => Ok(Self::Refinery),
            "miner" => Ok(Self::Miner),
            _ => Err(()),
        }
    }
}

impl Display for MachineKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            Self::Smelter  => write!(f, "smelter"),
            Self::Factory  => write!(f, "factory"),
            Self::Chemical => write!(f, "chemical"),
            Self::Refinery => write!(f, "refinery"),
            Self::Miner    => write!(f, "miner"),
        }
    }
}

// TODO: add a hash map struct for linking the products with a row in the csv file

#[derive(Debug, Clone, Default)]
struct Product {
    kind: ProductKind,
    time: f32,
    amount: i8,
    machine: MachineKind,
    recipe_products: Vec<RecipePart>,
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

impl PartialEq<String> for Product {
    fn eq(&self, other: &String) -> bool {
        self.kind.name == *other
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
    #[allow(dead_code)]
    fn new(kind: ProductKind, time: f32, amount: i8, crafter_kind: MachineKind, recipe_products: Vec<RecipePart>) -> Self {
        let product = Product{ kind, time, amount, machine: crafter_kind, recipe_products};
        product
    }
    fn from_lexer(tokens: &Vec<Token>) -> Self {
        if tokens.len() % 2 != 1 {
            println!("Error: Size of tokens was not the right length to create a product");
            return Product::default()
        } else if tokens.len() < 5 {
            println!("Error: Size of tokens was too short to create a product");
            return Product::default()
        }
        let recipe_products = RecipePart::parse_recipe(&tokens[5..]);

        return Product {
            kind: ProductKind{ name: tokens[1].text.clone() },
            time: tokens[2].text.parse().unwrap(),
            amount: tokens[3].text.parse().unwrap(),
            machine: tokens[4].text.parse().unwrap(),
            recipe_products
        }
    }
}

#[derive(PartialEq, Eq, Hash, Debug, Clone, Default)]
struct ProductKind {
    name: String,
}

impl ProductKind {
    #[allow(dead_code)]
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
        writeln!(f, "{}{}{}", color::Fg(color::Cyan), display, color::Fg(color::Reset))
    }
}

#[allow(dead_code)]
impl ProductList {
    fn get_product(&self, product_kind: &ProductKind) -> Option<&Product> {
        for p in &self.list {
            if p == product_kind {
                return Some(p)
            }
        }
        return None
    }
    fn get(&self, index: usize) -> &Product {
        return self.list.get(index).unwrap()
    }
    fn contains(&self, product_kind: &ProductKind) -> bool {
        return self.get_product(product_kind) != None
    }
    fn add(&mut self, product: Product) {
        self.list.push(product)
    }
    fn remove(&mut self, product: &Product) -> Product {
        let mut index = 0;
        for (i, p) in self.list.clone().into_iter().enumerate() {
            if p == product.clone() {
                index = i
            }
        }
        self.list.remove(index)
    }
    fn from_file_tokens(data: &Vec<LineData>) -> Self {
        let mut list = vec![];
        for line in data {
            match line.kind {
                DataKind::Setting    => {}
                DataKind::Product(_) => { list.push(Product::from_lexer(&line.tokens)) }
            }
        }
        return ProductList { list }
    }
}

struct Settings {
    speed: HashMap<MachineKind, f32>,
}

impl Settings {
    fn from_lexer(data: &Vec<LineData>) -> Self {
        let mut map: HashMap<MachineKind, f32> = HashMap::new();
        for line in data {
            match line.kind {
                DataKind::Setting    => { map.insert(line.tokens[1].text.parse().unwrap(), line.tokens[2].text.parse().unwrap()); }
                DataKind::Product(_) => {}
            }
        }
        return Settings { speed: map }
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

struct Tree {
    parent: Node,
    indent: usize,
    children: Vec<Box<Tree>>,
}

impl Display for Tree {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        if self.indent > 0 {
            let indentation = "    ".repeat(self.indent - 1 as usize);
            write!(f, "{}    {}", indentation, self.parent)
        } else {
            write!(f, "{}", self.parent)
        }
    }
}

impl Tree {
    fn new (node: Node, indent: usize, data: &ProductList, settings: &Settings) -> Tree {
        let mut children = vec![];
        let node_calc = generate_result(node.clone(), data, settings);
        for sub_node in node_calc {
            let boxed_sub_tree = Box::new(Tree::new(sub_node, indent + 1, data, settings));
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
    // TODO: make tree output interactive with tabs
    // TODO: add search function to tree to get the accumulated value for a certain product
}
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
enum TokenKind {
    Comment,
    Expr,
    Colon,
    Comma,
    Dash,
    DoubleDash,
    GreaterThan,
    NewLine,
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
    chars: Peekable<Char>,
    index: usize,
}

impl <Char: Iterator<Item=char>> Iterator for Lexer<Char> {
    type Item = Token;

    fn next(&mut self) -> Option<Self::Item> {
        let mut text = String::new();
        if let Some(x) = self.chars.next() {
            text.push(x);
            self.index += 1;
            return match x {
                ':' => Some(Token::new(TokenKind::Colon, text, self.index)),
                ',' => Some(Token::new(TokenKind::Comma, text, self.index)),
                '#' => Some(Token::new(TokenKind::Comment, text, self.index)),
                '-' => {
                    if self.chars.peek() == Some(&('-')) {
                        self.chars.next();
                        self.index += 1;
                        Some(Token::new(TokenKind::DoubleDash, text, self.index))
                    } else {
                        Some(Token::new(TokenKind::Dash, text, self.index))
                    }
                },
                ' '  => self.next(),
                '\n' => Some(Token::new(TokenKind::NewLine, text, self.index)),
                '>'  => Some(Token::new(TokenKind::GreaterThan, text, self.index)),
                char if char.is_alphanumeric() => {
                    #[allow(irrefutable_let_patterns)]
                    while let char = self.chars.peek().unwrap() {
                        if char.is_alphanumeric() || char.clone() == '_' || char.clone() == '.'{
                            text.push(self.chars.next().unwrap());
                            self.index += 1;
                        } else { break }
                    };
                    return Some(Token::new(TokenKind::Expr, text, self.index))
                }
                char => { SyntaxError(self.input.clone(), self.index, format!("Unexpected Character {}", char)).show(); None}
            }
        } else { None }
    }
}

impl <Char: Iterator<Item=char>> Lexer<Char> {
    fn new(input: String, chars: Char) -> Self{
        Self { input, chars: chars.peekable(), index: 0 }
    }
}

struct MatchInput {
    index: usize,
}

impl MatchInput {
    fn new() -> Self {
        return MatchInput{index: 0}
    }
    fn find (&mut self, input: String, data: Box<ProductList>) -> Option<(ProductKind, usize)> {
        if input == "" { return None}
        let mut temp_index = self.index;
        // TODO: factor out these two loops into 1 function
        for product in data.clone().list.into_iter() {
            if product.kind.name.starts_with(&input) {
                if temp_index > 0 { temp_index -= 1 }
                else { return Some((product.kind, 0)) }
            }
        }
        for product in data.clone().list {
            if let Some(place) = product.kind.name.find(&input) {
                if !(product.kind.name.starts_with(&input)) {
                    if temp_index > 0 { temp_index -= 1 }
                    else { return Some((product.kind, place)) }
                }
            }
        }
        if temp_index > 0 {
            self.index = 0;
            return self.find(input, data)
        }
        return None
    }
    fn _next (&mut self, _input: String, _data: &ProductList) {
    }
}

#[allow(dead_code)]
fn get_input(input_hint: String, strip_nl: bool) -> String {
    let mut input = String::new();

    print!("{}", input_hint);
    stdout().flush().expect("IOError could not flush stdout");
    stdin().read_line(&mut input).expect("IOError could not read stdin");

    if strip_nl && input.len() > 1 {
        input.drain((input.len() - 1)..);
    }

    return input
}

#[allow(dead_code)]
fn get_input_f32(input_hint: String) -> f32 {
    let mut float = String::new();

    print!("{}", input_hint);
    stdout().flush().expect("IOError could not flush stdout");
    stdin().read_line(&mut float).expect("IOError could not read stdin");

    let parsed_f32 = match float.clone().drain(..(float.len() - 1)).as_str().parse::<f32>() {
        Ok(out) => out,
        Err(_) => { ValueError(float, 9, format!("Could not interpret this as a number defaulting to 1.0")).show(); 1.0}
    };
    return parsed_f32
}

#[allow(dead_code)]
fn get_input_i16(input_hint: String) -> i16 {
    let mut int = String::new();

    print!("{}", input_hint);
    stdout().flush().expect("IOError could not flush stdout");
    stdin().read_line(&mut int).expect("IOError could not read stdin");

    let parsed_i16 = match int.clone().drain(..(int.len() - 1)).as_str().parse::<i16>() {
        Ok(out) => out,
        Err(_) => { ValueError(int, 9, format!("Could not interpret this as a number defaulting to 1")).show(); 1}
    };
    return parsed_i16
}

fn save_product_to_file(product: Product, file: String) {
    let mut line = format!("> {},{},{},{}", product.kind.name, product.time, product.amount, product.machine);
    for rec_part in product.recipe_products {
        line.push_str(&format!(",{}", rec_part))
    }
    let mut file = fs::OpenOptions::new()
        .append(true)
        .open(file)
        .unwrap();

    if let Err(e) = write!(file, "{}", line) {
        eprintln!("Couldn't write to file: {}", e);
    }
}

fn generate_result(node: Node, data: &ProductList, settings: &Settings) -> Vec<Node> {
    let mut result_vec = vec![];
    let sub = data.get_product(&node.product_kind);
    match sub {
        None => {
            ValueError( format!("{}", node.to_string()), 0, format!("When calculating {} was found in the recipe but not in 'products.csv'", node.to_string())).show()
        },
        Some(product) => {
            // multiplier is the machine type multiplier defined in products.csv
            let multiplier = settings.speed[&product.machine];
            // scalar references items per second
            let scalar = node.amount / product.time as f32 * multiplier;
            for recipe_part in product.recipe_products.clone() {
                let sub_time = data.get_product(&recipe_part.kind);
                match sub_time {
                    Some(sub_product) => {
                        result_vec.push(
                            Node {
                                product_kind: recipe_part.kind,
                                amount: scalar / settings.speed[&sub_product.machine] * recipe_part.amount as f32 * (sub_product.time / (sub_product.amount as f32)),
                            }
                        )
                    },
                    None => {
                        ValueError(
                            format!("{}", recipe_part.kind), 0,
                            format!("When calculating '{}', '{}' was found in the recipe but not in 'products.csv', ignoring this product", node.product_kind, recipe_part.kind)
                        ).show()
                    }
                }
            }
        }
    }
    result_vec
}

#[derive(PartialEq, Eq, Debug)]
enum DataKind {
    Setting,
    Product(ProductKind),
}

#[derive(Debug)]
#[allow(dead_code)]
struct LineData {
    row: i8,
    kind: DataKind,
    tokens: Vec<Token>,
}

fn lexing_file(filename: &str) -> Vec<LineData> {
    let mut file_data = vec![];
    let file = fs::read_to_string(filename).expect("failed to read file");
    let mut file_lexer = Lexer::new( file.clone(), file.chars() );

    // values needed for looping over the file data and storing them in a LineData struct
    let mut parsing = true;
    let mut tokens = Vec::<Token>::new();
    let mut row = 0;
    while parsing {
        match file_lexer.next() {
            None => { parsing = false }
            Some(token) => {
                match token.kind {
                    TokenKind::NewLine => {
                        if !tokens.is_empty() {
                            match tokens[0].kind {
                                TokenKind::Comment     => {}
                                TokenKind::GreaterThan => { file_data.push(LineData { row, kind: DataKind::Product(ProductKind{ name: tokens[1].text.clone() }), tokens }) }
                                TokenKind::DoubleDash  => { file_data.push(LineData { row, kind: DataKind::Setting, tokens }) }
                                _                      => ParsingError(tokens[0].text.clone(), 0, format!("Unexpected Token in products.csv")).raise()
                            }
                        }
                        row += 1;
                        tokens = vec![];
                    }
                    TokenKind::Comment     => tokens.push(token),
                    TokenKind::GreaterThan => tokens.push(token),
                    TokenKind::Expr        => tokens.push(token),
                    TokenKind::DoubleDash  => tokens.push(token),
                    TokenKind::Comma       => {}
                    token => println!("Unexpected Token {:?} in products.csv", token),
                }
            }
        }
    }
    return file_data
}

fn parse_lexer(lexer: &mut Lexer<Chars<'_>>) -> Option<Command> {
    if let Some(token) = lexer.next() {
        match token.kind {
            TokenKind::Expr => {
                match token.text.to_ascii_lowercase().as_str() {
                    "exit" => exit(0),
                    "new"  => {
                        let args:Vec<Token> = lexer.collect();
                        Some(Command { kind:CommandKind::New, args })
                    },
                    "calc" => {
                        let args:Vec<Token> = lexer.collect();
                        Some(Command { kind:CommandKind::Calc, args })
                    },
                    "help" => { Some(Command { kind:CommandKind::Help, args: vec![] }) },
                    "list" => { Some(Command { kind:CommandKind::List, args: vec![] }) },
                    "edit" => { Some(Command { kind:CommandKind::Edit, args: vec![] }) },
                    err => {
                        KeyError(err.to_string(), token.loc + 2, format!("Unexpected Command '{}' use Help for possible commands", err)).show();
                        None
                    }
                }
            },
            _ => { SyntaxError(token.clone().text, token.loc + 2, format!("Unexpected Token '{}' use Help for possible commands", token.text)); None },
        }
    } else { None }
}

struct InteractiveProductTerm {
    input: String,
    offset: usize,
    offset_text: String,
    is_running: bool,
    search_data: Box<ProductList>
}

#[allow(dead_code)]
impl InteractiveProductTerm {
    fn new(offset_text: String, data: &ProductList) -> Self {
        return InteractiveProductTerm {
            input: "".to_string(),
            offset: offset_text.len() + 1,
            offset_text,
            is_running: false,
            search_data: Box::new(data.clone())
        }
    }
    fn reset(&mut self, new_text: String) {
        self.input       = "".to_string();
        self.offset      = new_text.len() + 1;
        self.offset_text = new_text;
    }
    fn input_get_product(&self) -> Option<&Product> {
        self.search_data.get_product( &ProductKind { name: self.input.clone() } )
    }
    fn start_session(&mut self) {
        // TODO: refactor this section so it supports the interactive edit session
        let mut in_match = MatchInput::new();
        self.is_running = true;
        print!("{}", self.offset_text);
        stdout().flush().expect("IOError could not flush stdout");
        while self.is_running {
            let mut stdout = stdout().into_raw_mode().unwrap();
            let (_, y) = match stdout.cursor_pos() {
                Ok(pos) => pos,
                Err(err) => {
                    IOError( err.to_string(), 1, format!("Could not locate cursor, reverting to default 1, 1")).show();
                    break
                }
            };
            match in_match.find(self.input.clone(), self.search_data.clone()) {
                Some((pkind, index)) => {
                    print!(
                        "{}{}{}{}",
                        Goto(self.offset as u16, y),
                        color::Fg(color::Rgb(0x77, 0x77, 0x77)),
                        clear::AfterCursor,
                        pkind
                    );
                    print!("{}{}{}{}", Goto((index + self.offset) as u16, y), color::Fg(color::Green), self.input, color::Fg(color::Reset));
                }
                None if self.input == "" => { print!("{}{}product{}{}", Goto(self.offset as u16, y), color::Fg(color::Rgb(0x77, 0x77, 0x77)), clear::AfterCursor, Goto(self.offset as u16, y)) }
                None => {
                    print!("{}{}", Goto(self.offset as u16, y), clear::AfterCursor);
                    print!("{}{}{}{}", Goto((self.offset) as u16, y), color::Fg(color::Red), self.input, color::Fg(color::Reset));
                }
            };
            stdout.flush().unwrap();
            let key = parse_terminal();
            match key {
                Some(Key::Backspace) => {
                    if self.input.is_empty() {}
                    else { self.input.remove(self.input.len() - 1); }
                }
                Some(Key::Char('\n')) => {
                    if self.search_data.contains(&ProductKind {name: self.input.clone()}) {
                        self.is_running = false;
                    } else {
                        match in_match.find(self.input.clone(), self.search_data.clone()) {
                            Some((pkind, _)) => { self.input = pkind.name; self.is_running = false }
                            None => {}
                        };
                    }
                }
                Some(Key::Char('\t')) => {
                    self.input = match in_match.find(self.input.clone(), self.search_data.clone()) {
                        Some((pkind, _)) => {pkind.name}
                        None => self.input.clone()
                    };
                    in_match.index = 0;
                }
                Some(Key::Up)   => { if in_match.index > 0 { in_match.index -= 1 } }
                Some(Key::Down) => { in_match.index += 1 }
                Some(Key::Char(char)) => {
                    self.input.push(char)
                }
                _ => { break }
            }
        }
    }
}

struct InteractiveEditTerm {
    rows: Vec<EditRow>,
    selected_row: i16,
    is_running: bool,
    escape_pressed: bool,
    cursor_position: (u16, u16)
}

impl InteractiveEditTerm {
    fn new(product: Product, mode: i8) -> Self {
        // mode 0: New
        //      1: Edit
        if mode == 1 {
            print!("{}{}{}Editing: {}{}\n", clear::All, Goto(1, 1), style::Underline, product.kind.name, style::Reset);
        } else if mode == 0 {
            print!("{}{}{}New Product{}\n", clear::All, Goto(1, 1), style::Underline, style::Reset);
        }

        let mut header = vec!["Name".to_string(), "Time".to_string(), "Amount".to_string(), "Machine".to_string()];
        let mut values = vec![product.kind.name.clone(), product.time.to_string(), product.amount.to_string(), product.machine.to_string()];

        for rec_part in product.recipe_products.clone() {
            header.push(rec_part.kind.name);
            values.push(rec_part.amount.to_string())
        }
        let terminal_space = "\n".repeat(header.len() - 1);
        println!("{}", terminal_space);
        let mut stdout = stdout().into_raw_mode().unwrap();
        let (_, y) = match stdout.cursor_pos() {
            Ok(pos) => pos,
            Err(err) => {
                IOError( err.to_string(), 1, format!("Could not locate cursor, reverting to default 1, 1")).show();
                (1, 1)
            }
        };
        let mut width = 0;
        for head in header.clone() {
            width = max(width, head.len() + 4)
        }
        let mut rows = vec![];
        for (index, head) in header.into_iter().enumerate() {
            let value = values[index].clone();
            if index <= 3 {
                rows.push(EditRow::new(y - values.len() as u16 + index as u16, false, head, value, width as u16));
            } else {
                rows.push(EditRow::new(y - values.len() as u16 + index as u16, true, head, value, width as u16));
            }
        }
        return InteractiveEditTerm {
            rows,
            selected_row: 0,
            is_running: false,
            escape_pressed: false,
            cursor_position: (1, 1)
        }
    }
    fn start_session(&mut self, data: &ProductList) {
        self.is_running = true;
        for row in &self.rows {
            row.print(false)
        }
        let mut current_row = &mut self.rows[self.selected_row as usize];
        current_row.select(false);
        let mut stdout = stdout().into_raw_mode().unwrap();
        while self.is_running {
            self.cursor_position = match stdout.cursor_pos() {
                Ok(pos) => { pos }
                Err(_) => { self.cursor_position }
            };
            current_row = &mut self.rows[self.selected_row as usize];
            let key = parse_terminal();

            match key {
                Some(Key::Backspace)  => {
                    if self.cursor_position.0 > current_row.spacing {
                        current_row.value.remove((self.cursor_position.0 - current_row.spacing - 1) as usize);
                        print!("{}", cursor::Left(1));
                        current_row.select(true);
                        self.cursor_position.0 -= 1
                    }
                }
                Some(Key::Char('\n')) => {
                    self.is_running = false;
                    // if any errors are present in the editor set running to true again
                    for row in &self.rows {
                        if row.error {
                            self.is_running = true
                        }
                    }
                }
                Some(Key::Char('\t')) => {
                    if current_row.row_index > 4 {
                        current_row.change_recipe_head(data);
                    }
                }
                Some(Key::Up)   => {
                    current_row.deselect();
                    self.selected_row = (self.selected_row - 1).rem_euclid(self.rows.len() as i16);
                    let current_row = &mut self.rows[self.selected_row as usize];
                    current_row.select(false);
                }
                Some(Key::Down) => {
                    current_row.deselect();
                    self.selected_row = (self.selected_row + 1).rem_euclid(self.rows.len() as i16);
                    let current_row = &mut self.rows[self.selected_row as usize];
                    current_row.select(false);
                }
                Some(Key::Left)  => {
                    // checking whether the cursor can go left
                    if self.cursor_position.0 > current_row.spacing {
                        print!("{}", cursor::Left(1));
                        current_row.select(true);
                        self.cursor_position.0 -= 1
                    }
                }
                Some(Key::Right) => {
                    // checking whether the cursor can go right
                    if self.cursor_position.0 < current_row.spacing + current_row.value.len() as u16 {
                        print!("{}", cursor::Right(1));
                        current_row.select(true);
                        self.cursor_position.0 += 1
                    }
                }
                Some(Key::Char(char)) => {
                    current_row.value.insert((self.cursor_position.0 - current_row.spacing) as usize, char);
                    print!("{}", cursor::Right(1));
                    current_row.select(true);
                    self.cursor_position.0 += 1
                }
                Some(Key::Esc) => {
                    self.escape_pressed = true;
                    self.is_running = false;
                }
                Some(Key::Insert) => {
                    let row_index = self.rows[self.rows.len() - 1].row_index + 1;
                    let width     = self.rows[self.rows.len() - 1].spacing;
                    let new_row = EditRow::new(row_index, true, "placeholder".to_string(), "1".to_string(), width as u16);
                    self.rows.push(new_row.clone());
                    if width < ("placeholder".len() + 4) as u16 {
                        let mut new_rows = vec![];
                        for mut row in self.rows.clone() {
                            row.set_spacing(("placeholder".len() + 4) as u16);
                            row.print(false);
                            new_rows.push(row);
                        }
                        print!("\n{}", cursor::Up(1));
                        self.rows = new_rows
                    } else {
                        new_row.print(true);
                    }
                }
                _ => { self.escape_pressed = true; break }
            }
        }
        self.rows[self.rows.len() - 1].print(false);
        println!();
    }
    fn get_product(&self) -> Product {
        let mut recipe_products = vec![];
        for (index, rec_part_name) in self.rows[4..].into_iter().enumerate() {
            recipe_products.push( RecipePart { kind: ProductKind { name: rec_part_name.head.clone() }, amount: self.rows[index + 4].value.parse().unwrap() } )
        }
        println!();
        return Product {
            kind: ProductKind { name: self.rows[0].value.clone() },
            time: self.rows[1].value.parse().unwrap(),
            amount: self.rows[2].value.parse().unwrap(),
            machine: self.rows[3].value.parse().unwrap(),
            recipe_products
        }
    }
}

#[derive(Clone)]
struct EditRow {
    row_index: u16,
    is_recipe: bool,
    head: String,
    value: String,
    value_type: i8,
    spacing: u16,
    is_selected: bool,
    error: bool,
}

impl EditRow {
    fn new(row_index: u16, is_recipe: bool, head: String, value: String, spacing: u16) -> Self {
        /*  value type
            0: string
            1: integer
            2: float    */
        let value_type: i8;
        if head == "Name" {
            value_type = 0
        } else if head == "Time" {
            value_type = 1
        } else if head == "Machine" {
            value_type = 2
        } else {
            value_type = 3
        }
        return EditRow { row_index, is_recipe, head: head.clone(), value: value.clone(), value_type, spacing, is_selected: false, error: false }
    }
    fn print(&self, save_pos: bool) {
        if save_pos {
            print!("{}", cursor::Save)
        }
        print!("{}{}", Goto(0, self.row_index), clear::CurrentLine);
        if self.is_recipe {
            print!("{}", color::Fg(color::Green))
        }
        print!("{}{}", self.head, Goto(self.spacing, self.row_index));
        if self.error {
            print!("{}", color::Fg(color::Red));
        } else if self.is_selected {
            print!("{}", color::Fg(color::Yellow));
        } else {
            print!("{}", color::Fg(color::Green));
        }
        print!("{}{}", self.value, color::Fg(color::Reset));
        if save_pos {
            print!("{}", cursor::Restore);
        }
        stdout().flush().unwrap();
    }
    fn select(&mut self, save_cursor_pos: bool) {
        match self.value_type {
            0 => match self.value.parse::<String>() {
                Ok(_)  => self.error = false,
                Err(_) => self.error = true,
            }
            1 => match self.value.parse::<f32>() {
                Ok(_)  => self.error = false,
                Err(_) => self.error = true,
            }
            2 => match self.value.parse::<MachineKind>() {
                Ok(_)  => self.error = false,
                Err(_) => self.error = true,
            },
            3 => match self.value.parse::<i16>() {
                Ok(_)  => self.error = false,
                Err(_) => self.error = true,
            },
            _ => {}
        }
        self.is_selected = true;
        self.print(save_cursor_pos);
    }
    fn deselect(&mut self) {
        self.is_selected = false;
        self.print(false);
    }
    fn set_spacing(&mut self, width: u16) {
        self.spacing = width
    }
    fn change_recipe_head(&mut self, data: &ProductList) {
        let mut product_session = InteractiveProductTerm::new("".to_string(), data);

        if data.contains(&ProductKind { name: self.head.clone() }) {
            print!("{}{}", Goto((self.head.len() + 1) as u16, self.row_index as u16), clear::CurrentLine);
            product_session.input = self.head.clone();
        } else {
            print!("{}{}", Goto(1, self.row_index as u16), clear::CurrentLine)
        }
        product_session.start_session();
        match product_session.input_get_product() {
            None => {}
            Some(product) => { self.head = product.kind.name.clone() }
        };
        self.select(false);
        // TODO: reprint every recipe subproduct after changing product head
    }
}

fn parse_terminal() -> Option<Key> {
    let stdin = stdin();
    let _stdout = stdout().into_raw_mode().unwrap();
    for c in stdin.events() {
        let evt = c.unwrap();
        return match evt {
            Event::Key(key) => {
                Some(key)
            }
            _ => { None }
        }
    }
    return None
}

fn new_product_recipe_dialog() -> Option<(String, i8)> {
    let mut io_input = String::new();
    println!("recipe part:: ");
    print!("product >>");
    stdout().flush().expect("ERROR: Failed to print io::stdout buffer");
    stdin().read_line(&mut io_input).expect("ERROR: Failed to read io::stdin");
    let mut lexer = Lexer::new(io_input.clone(), io_input.chars());

    return match lexer.next() {
        Some(token) if token.kind == TokenKind::Expr => {
            match lexer.next() {
                Some(amount) => {
                    match amount.clone().text.parse::<i8>() {
                        Ok(int) => { Some((token.text, int)) }
                        Err(_) => {
                            ValueError(amount.text, amount.loc + 13, format!("Expected a number")).show();
                            None
                        }
                    }
                }
                _ => None
            }
        }
        _ => None
    }
}

fn main() {
    let file_lexer = lexing_file("products.csv");
    let mut data = ProductList::from_file_tokens(&file_lexer);
    let settings = Settings::from_lexer(&file_lexer);
    // print!("{:?}", data);
    println!("------------------------------------------------------");
    println!("Type Calc without arguments to get a guided calculation");
    println!("Type Help to get a list of all possible commands");
    println!("------------------------------------------------------");
    while !QUIT {
        let mut io_input = String::new();
        print!("\n> ");
        stdout().flush().expect("ERROR: Failed to print io::stdout buffer");
        stdin().read_line(&mut io_input).expect("ERROR: Failed to read io::stdin");
        let mut lexer = Lexer::new(io_input.clone(), io_input.chars());
        if let Some(command) = parse_lexer(&mut lexer) { command.run(&mut data, &settings) } else {};
    }
}