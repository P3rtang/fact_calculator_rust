use std::{fs, fmt::{Formatter, Display, Result}, str::FromStr, io::{stdout, stdin, Write}, cmp::{max, Ordering}, iter::Peekable, process::exit, str::Chars, collections::HashMap};
use termion::{color, event::{Event, Key, MouseEvent}, input::{TermRead}, raw::IntoRawMode, cursor, clear, style};
use termion::cursor::{DetectCursorPos, Goto};
use termion::input::MouseTerminal;
use crate::Error::{InputError, IOError, KeyError, ParsingError, SyntaxError, ValueError};

macro_rules! pkind {
    ($name:expr) => { ProductKind { name: $name.to_string() } }
}

#[derive(Debug, PartialEq)]
enum CommandKind {
    Help,
    Calc,
    List,
    New,
    Edit,
    Exit,
    // TODO: add Conf command (for basic settings like smelter speed)
}

struct Command {
    kind: CommandKind,
    args: Vec<Token>,
}

impl Command {
    fn run(&self, data: &mut ProductList, settings: &Settings, file_data: &mut FileData) {
        match &self.kind {
            CommandKind::Help    => println!("{}", self),
            CommandKind::Calc    => {
                match &self.args[..] {
                    [token1, token2] => {
                        let node = Node { product_kind: ProductKind { name: token1.clone().text }, amount: token2.text.parse().unwrap() };
                        let tree = Tree::new(node, 0, data, settings);
                        tree.traverse(None)
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
                        let mut tree = Tree::new(node, 0, data, settings);
                        tree.print(None);
                    }
                    [..] => {}
                }
            },
            CommandKind::List    => print!("{}", data),
            CommandKind::New     => {
                match &self.args[..] {
                    [Token{kind:TokenKind::NewLine, ..}] => {
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
                            file_data.add_line(new_product);
                            file_data.save_to_file();
                        }
                    }
                    [token, ..] => {
                        SyntaxError(token.clone().text, token.loc, format!("Incorrect use of the `new` command")).show()
                    }
                    _ => println!("Unreachable")
                }
            }
            CommandKind::Edit => {
                println!("Select a product to edit");

                let mut interactive_product_session = InteractiveProductTerm::new(format!("product >> "), data);
                interactive_product_session.start_session();
                if let Some(product) = interactive_product_session.input_get_product() {
                    let mut interactive_edit_session = InteractiveEditTerm::new(product.clone(), 1);
                    interactive_edit_session.start_session(data);
                    if !interactive_edit_session.escape_pressed {
                        let edited_product = interactive_edit_session.get_product();
                        data.remove(product);
                        data.add(edited_product.clone());
                        file_data.edit_product_line(product, edited_product);
                        file_data.save_to_file();
                    }
                } else {
                    println!();
                    InputError("".to_string(), 9, format!("Could not find {}", interactive_product_session.input)).show();
                }
            }
            CommandKind::Exit => {}
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
                writeln!(f, "{}^ SyntaxError: {}",  " ".repeat(*loc), info)
            }
            KeyError(_, loc, info)    => {
                writeln!(f, "{}^ KeyError: {}",     " ".repeat(*loc), info)
            }
            ValueError(_, loc, info)  => {
                writeln!(f, "{}^ ValueError: {}",   " ".repeat(*loc), info)
            }
            IOError(_, loc, info)  => {
                writeln!(f, "{}^ IOError: {}",      " ".repeat(*loc), info)
            }
            InputError(_, loc, info)  => {
                writeln!(f, "{}^ InputError: {}",   " ".repeat(*loc), info)
            }
            ParsingError(_, loc, info)  => {
                writeln!(f, "{}^ ParsingError: {}", " ".repeat(*loc), info)
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
    fn new(kind: String, time: f32, amount: i8, crafter_kind: MachineKind, recipe_products: Vec<RecipePart>) -> Self {
        let product = Product{ kind: pkind!(kind), time, amount, machine: crafter_kind, recipe_products};
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
            kind: pkind!(tokens[1].text),
            time: tokens[2].text.parse().unwrap(),
            amount: tokens[3].text.parse().unwrap(),
            machine: tokens[4].text.parse().unwrap(),
            recipe_products
        }
    }
    fn to_tokens(&self) -> Vec<Token> {
        let mut return_vec = vec![Token { kind: TokenKind::GreaterThan, text: ">".to_string(), loc: 0}];
        let chars = self.to_string(' ');
        let lexer = Lexer::new(chars.chars());
        let mut vec_lexer = lexer.collect::<Vec<Token>>();
        return_vec.append(&mut vec_lexer);
        return_vec
    }
    #[allow(dead_code)]
    fn to_string(&self, delim: char) -> String {
        let mut return_str = String::new();

        return_str.push_str(self.kind.name.as_str());
        return_str.push(delim);
        return_str.push_str(self.time.to_string().as_str());
        return_str.push(delim);
        return_str.push_str(self.amount.to_string().as_str());
        return_str.push(delim);
        return_str.push_str(self.machine.to_string().as_str());
        return_str.push(delim);
        for r_part in &self.recipe_products {
            return_str.push_str(r_part.kind.name.as_str());
            return_str.push(delim);
            return_str.push_str(r_part.amount.to_string().as_str());
            return_str.push(delim);
        }
        return return_str
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
                LineKind::Product(_) => { list.push(Product::from_lexer(&line.tokens)) }
                _ => {}
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
                LineKind::Setting    => { map.insert(line.tokens[1].text.parse().unwrap(), line.tokens[2].text.parse().unwrap()); }
                _ => {}
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

impl Node {
    fn generate_result(&self, product_data: &ProductList, settings: &Settings) -> Vec<Node> {
        let mut result_vec = vec![];
        let sub = product_data.get_product(&self.product_kind);
        match sub {
            None => {
                ValueError( format!("{}", self.to_string()), 0, format!("When calculating {} was found in the recipe but not in 'products.csv'", self
                    .to_string())).show()
            },
            Some(product) => {
                // multiplier is the machine type multiplier defined in products.csv
                let multiplier = settings.speed[&product.machine];
                // scalar references items per second
                let scalar = self.amount / product.time as f32 * multiplier;
                for recipe_part in product.recipe_products.clone() {
                    let sub_time = product_data.get_product(&recipe_part.kind);
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
                                format!("When calculating '{}', '{}' was found in the recipe but not in 'products.csv', ignoring this product", self
                                    .product_kind, recipe_part.kind)
                            ).show()
                        }
                    }
                }
            }
        }
        result_vec
    }
}

struct Tree {
    parent: Node,
    indent: usize,
    children: Vec<Box<Tree>>,
}

#[allow(dead_code)]
impl Tree {
    fn new (node: Node, indent: usize, data: &ProductList, settings: &Settings) -> Tree {
        let mut children = vec![];
        let node_calc = node.generate_result(data, settings);
        for sub_node in node_calc {
            let boxed_sub_tree = Box::new(Tree::new(sub_node, indent + 3, data, settings));
            children.push(boxed_sub_tree)
        }
        let tree = Tree { parent: node, indent, children };
        return tree
    }
    fn traverse (&self, depth: Option<u16>) {
        let mut stdout = stdout().into_raw_mode().unwrap();
        let start_position = stdout.cursor_pos().unwrap().1;
        let indentation = " ".repeat(self.indent as usize);
        println!("{}{}{}>{} {}", Goto(1, start_position), indentation, color::Fg(color::Magenta), color::Fg(color::Reset), self.parent);
        if let Some(depth) = depth {
            if depth > 0 {
                for node in &self.children {
                    node.traverse(Some(depth - 1))
                }
            }
        } else {
            for node in &self.children {
                node.traverse(None)
            }
        }
    }
    fn start_session(&mut self) {
        let stdin = stdin();
        let mut stdout = MouseTerminal::from(stdout().into_raw_mode().unwrap());
        stdout.flush().unwrap();

        self.print(None);

        for c in stdin.events() {
            let evt = c.unwrap();
            match evt {
                Event::Key(Key::Esc) => break,
                Event::Mouse(me) => {
                    match me {
                        MouseEvent::Release(_, _) => {
                            
                        },
                        _ => (),
                    }
                }
                _ => {}
            }
            stdout.flush().unwrap();
        }
    }
    fn hide_all(&self) {

    }
    fn print(&mut self, depth: Option<u16>) {
        print!("{}{}", clear::All, Goto(1, 1));
        stdout().flush().unwrap();
        self.traverse(depth)
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
                        text.push('-');
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
                    while let Some(char) = self.chars.next_if(|char| { char.is_alphanumeric() || char.clone() == '_' || char.clone() == '.' } ) {
                        text.push(char);
                        self.index += 1;
                    };
                    return Some(Token::new(TokenKind::Expr, text, self.index))
                }
                char => { SyntaxError(char.to_string(), self.index, format!("Unexpected Character {}", char)).show(); None}
            }
        } else { None }
    }
}

impl <Char: Iterator<Item=char>> Lexer<Char> {
    fn new(chars: Char) -> Self{
        Self { chars: chars.peekable(), index: 0 }
    }
}

struct MatchInput {
    index: usize,
    data: Box<ProductList>
}

impl MatchInput {
    fn new(data: Box<ProductList>) -> Self {
        return MatchInput{index: 0, data}
    }
    fn find (&mut self, input: &String) -> Option<(ProductKind, usize)> {
        if input == "" { return None}
        let mut temp_index = self.index;
        // TODO: factor out these two loops into 1 function
        for product in self.data.list.clone().into_iter() {
            if product.kind.name.starts_with(input) {
                if temp_index > 0 { temp_index -= 1 }
                else { return Some((product.kind, 0)) }
            }
        }
        for product in self.data.list.clone() {
            if let Some(place) = product.kind.name.find(input) {
                if !(product.kind.name.starts_with(input)) {
                    if temp_index > 0 { temp_index -= 1 }
                    else { return Some((product.kind, place)) }
                }
            }
        }
        if temp_index > 0 {
            self.index = 0;
            return self.find(input)
        }
        return None
    }
    fn _next (&mut self, _input: String, _data: &ProductList) {
    }
}

#[derive(PartialEq, Eq, Debug, Clone)]
enum LineKind {
    Setting,
    Product(ProductKind),
    Comment(String),
    Empty,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct LineData {
    row: i8,
    kind: LineKind,
    tokens: Vec<Token>,
}

impl LineData {
    fn from_product(row: i8, product: Product) -> Self {
        return LineData { row, kind: LineKind::Product(product.kind.clone()), tokens: product.to_tokens() }
    }
}

#[derive(Clone)]
struct FileData {
    filename: String,
    line_data: Vec<LineData>,
}

impl FileData {
    fn save_to_file(&self) {
        let file_data = self.create_string_from_tokens();

        let mut file = fs::OpenOptions::new().write(true).open(&self.filename).expect("failed to open file");
        write!(file, "{}", file_data).expect("failed to write")
    }
    fn create_string_from_tokens(&self) -> String {
        let mut return_str = "".to_string();

        for line in &self.line_data {
            for token in &line.tokens {
                return_str.push_str(token.text.as_str());
                match token.kind {
                    TokenKind::GreaterThan => return_str.push(' '),
                    TokenKind::DoubleDash  => return_str.push(' '),
                    TokenKind::Comment     => return_str.push(' '),
                    _                      => return_str.push(','),
                }
            }
            return_str.push('\n');
        }
        return return_str
    }
    fn edit_product_line(&mut self, old_prod: &Product, new_prod: Product) {
        if let Some(index) = self.get_line_index(&old_prod.kind) {
            let _line = self.line_data.remove(index);
            self.line_data.push(LineData::from_product(self.line_data.len() as i8, new_prod))
        } else {}
    }
    fn get_line_index(&self, pkind: &ProductKind) -> Option<usize> {
        for (index, line) in self.line_data.clone().into_iter().enumerate() {
            match &line.kind {
                LineKind::Product(kind) if kind == pkind => { return Some(index) }
                _ => {}
            }
        }
        return None
    }
    fn add_line(&mut self, product: Product) {
        self.line_data.push(LineData::from_product(self.line_data.len() as i8, product))
    }
}

// TODO: add this in FileData struct
fn lexing_file(filename: &str) -> FileData {
    let mut line_data = vec![];
    let file = fs::read_to_string(filename).expect("failed to read file");
    let mut file_lexer = Lexer::new( file.chars() );

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
                                TokenKind::Comment     => {
                                    let mut comment = String::new();
                                    for token in &tokens {
                                        comment.push_str(format!(" {}", token.text).as_str())
                                    }
                                    line_data.push(LineData { row, kind: LineKind::Comment(comment), tokens})
                                }
                                TokenKind::GreaterThan => { line_data.push(LineData { row, kind: LineKind::Product(ProductKind{ name: tokens[1].text.clone() }), tokens }) }
                                TokenKind::DoubleDash  => { line_data.push(LineData { row, kind: LineKind::Setting, tokens }) }
                                _                      => ParsingError(tokens[0].text.clone(), 0, format!("Unexpected Token in products.csv")).raise()
                            }
                        } else {
                            line_data.push( LineData { row, kind: LineKind::Empty, tokens } )
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
    return FileData { filename: filename.to_string(), line_data }
}

fn parse_lexer(lexer: &mut Lexer<Chars<'_>>) -> Option<Command> {
    if let Some(token) = lexer.next() {
        match token.kind {
            TokenKind::Expr => {
                match token.text.to_ascii_lowercase().as_str() {
                    "exit" => Some(Command { kind:CommandKind::Exit, args: vec![] }),
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

fn parse_terminal() -> Option<Event> {
    let stdin = stdin();
    let _stdout = MouseTerminal::from(stdout().into_raw_mode().unwrap());
    for c in stdin.events() {
        let evt = c.unwrap();
        return Some(evt)
    }
    return None
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
    fn input_get_product(&self) -> Option<&Product> { self.search_data.get_product( &ProductKind { name: self.input.clone() } ) }
    fn start_session(&mut self) {
        // TODO: refactor this section so it supports the interactive edit session
        let mut in_match = MatchInput::new(self.search_data.clone());
        self.is_running = true;
        print!("{}", self.offset_text);
        stdout().flush().expect("IOError could not flush stdout");

        fn set_match_color(this: &mut InteractiveProductTerm, input_match: &mut MatchInput, y_pos: u16) {
            match input_match.find(&this.input) {
                Some((pkind, index)) => {
                    print!(
                        "{}{}{}{}",
                        Goto(this.offset as u16, y_pos),
                        color::Fg(color::Rgb(0x77, 0x77, 0x77)),
                        clear::AfterCursor,
                        pkind
                    );
                    print!("{}{}{}{}", Goto((index + this.offset) as u16, y_pos), color::Fg(color::Green), this.input, color::Fg(color::Reset));
                }
                None if this.input == "" => {
                    print!(
                        "{}{}product{}{}",
                        Goto(this.offset as u16, y_pos), color::Fg(color::Rgb(0x77, 0x77, 0x77)),
                        clear::AfterCursor, Goto(this.offset as u16, y_pos)
                    )
                }
                None => {
                    print!("{}{}", Goto(this.offset as u16, y_pos), clear::AfterCursor);
                    print!("{}{}{}{}", Goto((this.offset) as u16, y_pos), color::Fg(color::Red), this.input, color::Fg(color::Reset));
                }
            };
        }

        while self.is_running {
            let mut stdout = stdout().into_raw_mode().unwrap();
            let (_, y) = match stdout.cursor_pos() {
                Ok(pos) => pos,
                Err(err) => {
                    IOError( err.to_string(), 1, format!("Could not locate cursor, reverting to default 1, 1")).show();
                    break
                }
            };
            set_match_color(self, &mut in_match, y);
            stdout.flush().unwrap();
            let event = parse_terminal();
            match event {
                Some(Event::Key(key)) => match key {
                    Key::Backspace => {
                        if self.input.is_empty() {} else { self.input.remove(self.input.len() - 1); }
                    }
                    Key::Char('\n') => {
                        if self.search_data.contains(&ProductKind { name: self.input.clone() }) {
                            self.is_running = false;
                        } else {
                            match in_match.find(&self.input) {
                                Some((pkind, _)) => {
                                    self.input = pkind.name;
                                    set_match_color(self, &mut in_match, y);
                                    self.is_running = false;
                                }
                                None => {}
                            };
                        }
                    }
                    Key::Char('\t') => {
                        self.input = match in_match.find(&self.input) {
                            Some((pkind, _)) => { pkind.name }
                            None => self.input.clone()
                        };
                        // setting the pattern matching index back to zero
                        in_match.index = 0;
                    }
                    Key::Up => { if in_match.index > 0 { in_match.index -= 1 } }
                    Key::Down => { in_match.index += 1 }
                    Key::Char(char) => {
                        self.input.push(char)
                    }
                    _ => {}
                }
                _ => { break }
            }
        }
    }
}

#[derive(Default)]
struct InteractiveEditTerm {
    rows: Vec<EditRow>,
    selected_row: i16,
    is_running: bool,
    escape_pressed: bool,
    cursor_position: (u16, u16),
    column_width: Vec<usize>,
}

impl InteractiveEditTerm {
    fn new(product: Product, mode: i8) -> Self {
        // mode 0: New
        //      1: Edit
        print!("{}", clear::All);
        if mode == 1 {
            print!("{}{}{}Editing: {}{}\n", clear::All, Goto(1, 1), style::Underline, product.kind.name, style::Reset);
        } else if mode == 0 {
            print!("{}{}{}New Product{}\n", clear::All, Goto(1, 1), style::Underline, style::Reset);
        }

        let mut header = vec!["Name".to_string(), "Time".to_string(), "Amount".to_string(), "Machine".to_string()];
        let mut values = vec![product.kind.name.clone(), product.time.to_string(), product.amount.to_string(), product.machine.to_string()];
        let mut spacing = vec![10, max(values[0].len(), values[3].len()) + 3];

        for rec_part in product.recipe_products.clone() {
            header.push(rec_part.kind.name.clone());
            values.push(rec_part.amount.to_string());

            spacing = vec![max(spacing[0], rec_part.kind.name.len() + 3), max(spacing[1], rec_part.amount.to_string().len() + 3)];
        }

        let mut _stdout = stdout().into_raw_mode().unwrap();
        let mut rows = vec![];
        let mut row_types = vec![vec![0, 0], vec![0, 1], vec![0, 3], vec![0, 2]];
        for _ in 4..header.len() {
            row_types.push(vec![0, 3])
        }
        for (index, head) in header.into_iter().enumerate() {
            let value = values[index].clone();
            rows.push(
                EditRow::new(
                    (index + 2) as u16, true, head, vec![value], row_types[index].clone(), spacing.clone()
                )
            );
        }
        let mut term = InteractiveEditTerm::default();
        term.rows = rows;
        term.column_width = spacing;
        return term
    }
    fn start_session(&mut self, data: &ProductList) {
        self.is_running = true;
        self.print();

        let mut current_row = &mut self.rows[self.selected_row as usize];
        current_row.select(1, false);
        let mut stdout = stdout().into_raw_mode().unwrap();
        while self.is_running {
            self.cursor_position = match stdout.cursor_pos() {
                Ok(pos) => { pos }
                Err(_) => { self.cursor_position }
            };
            current_row = &mut self.rows[self.selected_row as usize];
            let event = parse_terminal();
            match event {
                Some(Event::Key(key)) => match key {
                    Key::Backspace  => {
                        if self.cursor_position.0 > current_row.body[0].len() as u16 {
                            current_row.body[0].remove((self.cursor_position.0 - current_row.spacing[0] as u16 - 1) as usize);
                            print!("{}", cursor::Left(1));
                            current_row.select(1, true);
                            self.cursor_position.0 -= 1
                        }
                    }
                    Key::Char('\n') => {
                        self.is_running = false;
                        // if any errors are present in the editor set running to true again
                        for row in &self.rows {
                            if row.error {
                                self.is_running = true
                            }
                        }
                    }
                    Key::Char('\t') => {
                        if current_row.row_index > 4 {
                            current_row.change_recipe_head(data);
                            self.print()
                        }
                    }
                    Key::Up   => {
                        current_row.deselect();
                        self.selected_row = (self.selected_row - 1).rem_euclid(self.rows.len() as i16);
                        let current_row = &mut self.rows[self.selected_row as usize];
                        current_row.select(1, false);
                    }
                    Key::Down => {
                        current_row.deselect();
                        self.selected_row = (self.selected_row + 1).rem_euclid(self.rows.len() as i16);
                        let current_row = &mut self.rows[self.selected_row as usize];
                        current_row.select(1, false);
                    }
                    Key::Left  => {
                        // checking whether the cursor can go left
                        if self.cursor_position.0 > current_row.spacing[0] as u16 {
                            print!("{}", cursor::Left(1));
                            current_row.select(1, true);
                            self.cursor_position.0 -= 1
                        }
                    }
                    Key::Right => {
                        // checking whether the cursor can go right
                        if self.cursor_position.0 < (current_row.spacing[0] + current_row.body.len()) as u16 {
                            print!("{}", cursor::Right(1));
                            current_row.select(1, true);
                            self.cursor_position.0 += 1
                        }
                    }
                    Key::Char(char) => {
                        current_row.body[0].insert(self.cursor_position.0 as usize - current_row.spacing[0], char);
                        print!("{}", cursor::Right(1));
                        current_row.select(1, true);
                        self.cursor_position.0 += 1
                    }
                    Key::Esc => {
                        self.escape_pressed = true;
                        self.is_running = false;
                    }
                    Key::Insert => {
                        let row_index = self.rows[self.rows.len() - 1].row_index + 1;
                        let new_row = EditRow::new(row_index, true, "placeholder".to_string(), vec!["1".to_string()], vec![0, 3], self.column_width.clone());
                        self.rows.push(new_row.clone());
                        self.column_width = self.get_column_width();
                        if self.rows[0].spacing != self.column_width {
                            for row in &mut self.rows {
                                row.spacing = self.column_width.clone()
                            }
                        }
                        self.print()
                    }
                    _ => { self.escape_pressed = true; break }
                }
                _ => {}
            }
        }
        self.rows[self.rows.len() - 1].print(false);
        println!();
    }
    fn print(&self) {
        for row in &self.rows {
            row.print(false)
        }
    }
    fn get_column_width(&self) -> Vec<usize> {
        let columns = self.rows[0].body.len() + 1;
        let mut max_width = vec![0; columns];
        for row in &self.rows {
            for i in 0..columns {
                max_width[i] = if i == 0 { max(max_width[i], row.head.len() + 3) } else { max(max_width[i], row.body[i - 1].len() + 3) }
            }
        }
        return max_width
    }
    fn get_product(&self) -> Product {
        let mut recipe_products = vec![];
        for (index, rec_part_name) in self.rows[4..].into_iter().enumerate() {
            recipe_products.push( RecipePart { kind: ProductKind { name: rec_part_name.head.clone() }, amount: self.rows[index + 4].body[0].parse().unwrap() } )
        }
        println!();
        return Product::new(self.rows[0].body[0].clone(), self.rows[1].body[0].parse().unwrap(), self.rows[2].body[0].parse().unwrap(),
                            self.rows[3].body[0].parse().unwrap(), recipe_products)
    }
}

#[derive(Clone)]
struct EditRow {
    row_index: u16,
    is_recipe: bool,
    head: String,
    body: Vec<String>,
    body_types: Vec<i8>,
    spacing: Vec<usize>,
    is_selected: bool,
    error: bool,
}

impl EditRow {
    fn new(row_index: u16, is_recipe: bool, head: String, body: Vec<String>, body_types: Vec<i8>, spacing: Vec<usize>) -> Self {
        return EditRow { row_index, is_recipe, head, body, body_types, spacing, is_selected: false, error: false }
    }
    fn print(&self, save_pos: bool) {
        if save_pos {
            print!("{}", cursor::Save)
        }
        print!("{}{}", Goto(1, self.row_index), clear::CurrentLine);
        if self.is_recipe {
            print!("{}", color::Fg(color::Green))
        }
        print!("{}{}", self.head, Goto(self.spacing[0] as u16, self.row_index));
        if self.error {
            print!("{}", color::Fg(color::Red));
        } else if self.is_selected {
            print!("{}", color::Fg(color::Yellow));
        } else {
            print!("{}", color::Fg(color::Green));
        }
        print!("{}{}", self.body[0], color::Fg(color::Reset));
        if save_pos {
            print!("{}", cursor::Restore);
        }
        stdout().flush().unwrap();
    }
    fn select(&mut self, column: usize, save_cursor_pos: bool) {
        match self.body_types[column] {
            0 => match self.body[0].parse::<String>() {
                Ok(_)  => self.error = false,
                Err(_) => self.error = true,
            }
            1 => match self.body[0].parse::<f32>() {
                Ok(_)  => self.error = false,
                Err(_) => self.error = true,
            }
            2 => match self.body[0].parse::<MachineKind>() {
                Ok(_)  => self.error = false,
                Err(_) => self.error = true,
            },
            3 => match self.body[0].parse::<i16>() {
                Ok(_)  => self.error = false,
                Err(_) => self.error = true,
            },
            _ => unreachable!()
        }
        self.is_selected = true;
        self.print(save_cursor_pos);
    }
    fn deselect(&mut self) {
        self.is_selected = false;
        self.print(false);
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
        self.select(1, false);
    }
}

fn main() {
    let mut file_lexer = lexing_file("products.csv");
    let mut data = ProductList::from_file_tokens(&file_lexer.line_data);
    let settings = Settings::from_lexer(&file_lexer.line_data);

    let mut quit = false;
    println!("------------------------------------------------------");
    println!("Type Calc without arguments to get a guided calculation");
    println!("Type Help to get a list of all possible commands");
    println!("------------------------------------------------------");
    while !quit {
        let mut io_input = String::new();
        print!("\n> ");
        stdout().flush().expect("ERROR: Failed to print io::stdout buffer");
        stdin().read_line(&mut io_input).expect("ERROR: Failed to read io::stdin");
        let mut lexer = Lexer::new(io_input.chars());
        if let Some(command) = parse_lexer(&mut lexer) {
            if command.kind == CommandKind::Exit {
                file_lexer.save_to_file();
                quit = true;
            }
            command.run(&mut data, &settings, &mut file_lexer)
        } else {};
    }
}