use std::{fs, fmt::{Formatter, Display, Result}, io::{stdout, stdin, Write}};
use std::cmp::Ordering;
use std::fs::OpenOptions;
use std::iter::{Enumerate, Peekable};
use std::process::exit;
use std::str::Chars;
use ansi_term::Colour;
use termion::{color, event::{Event, Key}, input::{TermRead}, raw::IntoRawMode, cursor::DetectCursorPos};
use crate::Error::{InputError, IOError, KeyError, SyntaxError, ValueError};

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
    fn run(&self, data: &mut ProductList) {
        match &self.kind {
            CommandKind::Help    => println!("{}", self),
            CommandKind::Calc    => {
                match &self.args[..] {
                    [token1, token2] => {
                        let node = Node { product_kind: ProductKind { name: token1.clone().text }, amount: token2.text.parse().unwrap() };
                        let tree = Tree::new(node, 0, data);
                        tree.traverse()
                    }
                    [err,..] => {
                        SyntaxError(err.clone().text, err.loc, format!("expected 2 arguments for calc found {}", self.args.len())).show()
                    }
                    [] => {
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
                        let tree = Tree::new(node, 0, data);
                        tree.traverse()
                    }
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
                        let new_product = Self::new_interactive_session(data);
                        println!("{:?}", new_product);
                        data.add(new_product.clone());
                        save_product_to_file(new_product, "products.csv".to_string())
                    }
                }
            }
            CommandKind::Edit => {
                let _edit_product = Self::edit_interactive_session(data);
                todo!("Edit not implemented yet")
            }
        }
    }
    fn new_interactive_session(data: &ProductList) -> Product {
        let name   = get_input    ("             name >> ".to_string(), true);
        let time   = get_input_f32("  production_time >> ".to_string());
        let amount = get_input_i16("production amount >> ".to_string()) as i8;

        println!("recipe parts: ");

        let mut recipe_products = vec![];
        let mut part_num = 1;
        let mut interactive_terminal = InteractiveProductTerm::new(format!("Part {} >> ", part_num), data);
        interactive_terminal.start_session();

        let mut product_name = interactive_terminal.input.clone();
        while product_name != "" {
            part_num += 1;
            interactive_terminal.reset(format!("Part {} >> ", part_num));
            let req_amount = get_input_i16(format!("\ncost >> ")) as i8;
            recipe_products.push( RecipePart { kind: ProductKind { name: product_name }, amount: req_amount } );
            interactive_terminal.start_session();
            product_name = interactive_terminal.input.clone();
        }

        return Product {
            kind: ProductKind { name },
            time,
            amount,
            recipe_products,
        }
    }
    fn edit_interactive_session(data: &ProductList) -> Product {
        println!("Select a product to edit");

        let mut interactive_product_session = InteractiveProductTerm::new(format!("product >> "), data);
        interactive_product_session.start_session();
        if let Some(product) = interactive_product_session.input_get_product() {
            let mut session_type = "".to_string();
            println!("\n(1) Edit the Product");
            println!("(2) Edit the Recipe");
            print!("Enter (1) or (2) >> ");
            stdout().flush().expect("IOError could not flush stdout");
            stdin().read_line(&mut session_type).expect("IOError could not read stdin");
            let mut interactive_edit_session = InteractiveEditTerm::new(session_type.replace("\n", ""), product.clone());
            interactive_edit_session.start_session();
        } else {
            println!();
            InputError( "".to_string(), 9, format!("Could not find {}", interactive_product_session.input)).show()
        }

        return Product {
            kind: ProductKind { name: "".to_string() },
            time: 0.0,
            amount: 0,
            recipe_products: vec![]
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

#[derive(Debug)]
enum Error {
    SyntaxError (String, usize, String),
    KeyError    (String, usize, String),
    ValueError  (String, usize, String),
    IOError     (String, usize, String),
    InputError  (String, usize, String),
}

impl Error {
    fn _raise(&self) {
        self.show();
        exit(1);
    }
    fn show(&self) {
        match self {
            SyntaxError(_, _, _) => { eprint!("{}", self) }
            KeyError   (_, _, _) => { eprint!("{}", self) }
            ValueError (_, _, _) => { eprint!("{}", self) }
            IOError    (_, _, _) => { eprint!("{}", self) }
            InputError (_, _, _) => { eprint!("{}", self) }
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
    fn parse_recipe(recipe_list: &[&str]) -> Vec<Self> {
        let mut return_map = vec![];
        for sub in recipe_list {
            let sub_vec = sub.split(':').collect::<Vec<&str>>();
            return_map.push(RecipePart { kind: ProductKind::new(sub_vec[0].to_string()), amount: sub_vec[1].parse().unwrap() });
        }
        return return_map
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct Product {
    kind: ProductKind,
    time: f32,
    amount: i8,
    // TODO: add production type (eg. smelter, factory, ...)
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
    fn new(kind: ProductKind, time: f32, amount: i8, recipe_products: Vec<RecipePart>) -> Product {
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

#[allow(dead_code)]
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

#[allow(dead_code)]
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
    // TODO: make tree output interactive with tabs
    // TODO: add search function to tree to get the accumulated value for a certain product
}

#[derive(Debug, Clone, PartialEq)]
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
                    #[allow(irrefutable_let_patterns)]
                    while let char = self.chars.peek().unwrap().1 {
                        if char.is_alphanumeric() || char == '_' {
                            text.push(self.chars.next().unwrap().1)
                        } else { break }
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

#[allow(dead_code)]
struct MatchInput {
    index: usize,
    match_index_list: Vec<usize>,
}

impl MatchInput {
    fn new() -> Self {
        return MatchInput{index: 0, match_index_list: Vec::new()}
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
    let mut line = format!("{},{},{}", product.kind.name, product.time, product.amount);
    for rec_part in product.recipe_products {
        line.push_str(&format!(",{}", rec_part))
    }
    let mut file = OpenOptions::new()
        .append(true)
        .open(file)
        .unwrap();

    if let Err(e) = write!(file, "{}", line) {
        eprintln!("Couldn't write to file: {}", e);
    }
}

fn generate_result(node: Node, data: &ProductList) -> Vec<Node> {
    let mut result_vec = vec![];
    // scalar references items per second
    let sub = data.get_product(&node.product_kind);
    match sub {
        None => {},
        Some(product) => {
            let scalar = node.amount / product.time as f32;
            for recipe_part in product.recipe_products.clone() {
                let sub_time = data.get_product(&recipe_part.kind);
                match sub_time {
                    Some(sub_product) => {
                        println!("{:?}", sub_product);
                        result_vec.push(
                            Node {
                                product_kind: recipe_part.kind,
                                amount: scalar as f32 * recipe_part.amount as f32 * (sub_product.time / (sub_product.amount as f32)),
                            }
                        )
                    },
                    None => {}
                }
            }
        }
    }
    result_vec
}

fn parse_file(filename: &str) -> ProductList{
    // TODO: report duplicate entries
    let mut list = vec![];

    let file = fs::read_to_string(filename)
        .expect("failed to read file");
    for line in file.split('\n') {
        match line.chars().peekable().peek() {
            Some('#') => { },
            _  => {
                let parse_line: Vec<&str> = line.split(',').collect();
                if parse_line != [""] {
                    let p = Product::new(
                        ProductKind::new(parse_line[0].to_string()),
                        parse_line[1].parse().unwrap(),
                        parse_line[2].parse().unwrap(),
                        RecipePart::parse_recipe(&parse_line[3..])
                    );
                    list.push(p)
                }
                else {}
            }
        }
    }
    return ProductList { list }
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
                        KeyError(err.to_string(), token.loc, "Unexpected Command use Help for possible commands".to_string()).show();
                        None
                    }
                }
            },
            _ => { SyntaxError(token.clone().text, token.loc, format!("Unexpected Token {} use Help for possible commands", token.text)); None },
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
        let mut in_match = MatchInput::new();
        self.is_running = true;
        print!("{}", self.offset_text);
        stdout().flush().expect("IOError could not flush stdout");
        while self.is_running {
            let key = parse_terminal();

            match key {
                Some(Key::Backspace) => {
                    if self.input.is_empty() {}
                    else { self.input.remove(self.input.len() - 1); }
                }
                Some(Key::Char('\n')) => { self.is_running = false }
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
                        termion::cursor::Goto(self.offset as u16, y),
                        color::Fg(color::Rgb(0x77, 0x77, 0x77)),
                        termion::clear::AfterCursor,
                        pkind
                    );
                    print!("{}{}{}{}", termion::cursor::Goto((index + self.offset) as u16, y), color::Fg(color::Green), self.input, color::Fg(color::Reset));
                }
                None if self.input == "" => { print!("{}{}", termion::cursor::Goto(self.offset as u16, y), termion::clear::AfterCursor) }
                None => {
                    print!("{}{}", termion::cursor::Goto(self.offset as u16, y), termion::clear::AfterCursor);
                    print!("{}{}{}{}", termion::cursor::Goto((self.offset) as u16, y), color::Fg(color::Red), self.input, color::Fg(color::Reset));
                }
            };
            stdout.flush().unwrap();
        }
    }
}

#[allow(dead_code)]
struct InteractiveEditTerm {
    header: Vec<String>,
    values: Vec<String>,
    edit_product: Product,
}

impl InteractiveEditTerm {
    fn new(edit_type: String, product: Product) -> Self {
        let mut header = vec![];
        let mut values = vec![];
        match edit_type.as_str() {
            "1" => {
                header = vec!["Name".to_string(), "Time".to_string(), "Amount".to_string()];
                values = vec![product.kind.name.clone(), product.time.to_string(), product.amount.to_string()];
            }
            "2" => {
                for rec_part in product.recipe_products.clone() {
                    header.push(rec_part.kind.name);
                    values.push(rec_part.amount.to_string())
                }
            }
            "" => {
                header = vec!["Name".to_string(), "Time".to_string(), "Amount".to_string()];
                values = vec![product.kind.name.clone(), product.time.to_string(), product.amount.to_string()];
            }
            rest => {
                InputError(rest.to_string(), 20, format!("Expected 1 or 2 but got {}", rest)).show();
            }
        }
        return InteractiveEditTerm {
            header,
            values,
            edit_product: product
        }
    }
    fn start_session(&mut self) {

    }
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
    let mut data = parse_file("products.csv");
    // print!("{:?}", data);
    println!("------------------------------------------------------");
    println!("Use the Calc command without arguments to get a guided calculation");
    println!("or give the name of the product and amount separated by a space");
    println!("example: 'calc green_circuit 10'");
    println!("------------------------------------------------------");
    while !QUIT {
        let mut io_input = String::new();
        print!("\n> ");
        stdout().flush().expect("ERROR: Failed to print io::stdout buffer");
        stdin().read_line(&mut io_input).expect("ERROR: Failed to read io::stdin");
        let mut lexer = Lexer::new(io_input.clone(), io_input.chars());
        if let Some(command) = parse_lexer(&mut lexer) { command.run(&mut data) } else {};
    }
}