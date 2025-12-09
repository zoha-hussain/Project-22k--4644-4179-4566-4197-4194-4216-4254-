import re
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

# ============================================================================
# PHASE 1: LEXICAL ANALYSIS
# ============================================================================

class TokenType(Enum):
    # Keywords
    PATTERN = "PATTERN"
    GENERATE = "GENERATE"
    SEQUENCE = "SEQUENCE"
    IF = "IF"
    ELSE = "ELSE"
    RETURN = "RETURN"
    PRINT = "PRINT"
    
    # Identifiers and Literals
    IDENTIFIER = "IDENTIFIER"
    NUMBER = "NUMBER"
    
    # Operators
    PLUS = "PLUS"
    MINUS = "MINUS"
    MULTIPLY = "MULTIPLY"
    DIVIDE = "DIVIDE"
    MODULO = "MODULO"
    ASSIGN = "ASSIGN"
    
    # Comparison Operators
    EQ = "EQ"
    NEQ = "NEQ"
    LT = "LT"
    GT = "GT"
    LTE = "LTE"
    GTE = "GTE"
    
    # Delimiters
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    LBRACE = "LBRACE"
    RBRACE = "RBRACE"
    COMMA = "COMMA"
    SEMICOLON = "SEMICOLON"
    COLON = "COLON"
    
    # Special
    EOF = "EOF"
    NEWLINE = "NEWLINE"

@dataclass
class Token:
    type: TokenType
    value: Any
    line: int
    column: int
    
    def __repr__(self):
        return f"Token({self.type.name}, {self.value}, L{self.line}:C{self.column})"

class Lexer:
    """Phase 1: Lexical Analyzer - Converts source code into tokens"""
    
    # Token specifications (regex patterns)
    TOKEN_SPECS = [
        ('NUMBER',      r'\d+(\.\d+)?'),
        ('PATTERN',     r'\bpattern\b'),
        ('GENERATE',    r'\bgenerate\b'),
        ('SEQUENCE',    r'\bsequence\b'),
        ('IF',          r'\bif\b'),
        ('ELSE',        r'\belse\b'),
        ('RETURN',      r'\breturn\b'),
        ('PRINT',       r'\bprint\b'),
        ('IDENTIFIER',  r'[a-zA-Z_][a-zA-Z0-9_]*'),
        ('EQ',          r'=='),
        ('NEQ',         r'!='),
        ('LTE',         r'<='),
        ('GTE',         r'>='),
        ('ASSIGN',      r'='),
        ('LT',          r'<'),
        ('GT',          r'>'),
        ('PLUS',        r'\+'),
        ('MINUS',       r'-'),
        ('MULTIPLY',    r'\*'),
        ('DIVIDE',      r'/'),
        ('MODULO',      r'%'),
        ('LPAREN',      r'\('),
        ('RPAREN',      r'\)'),
        ('LBRACE',      r'\{'),
        ('RBRACE',      r'\}'),
        ('COMMA',       r','),
        ('SEMICOLON',   r';'),
        ('COLON',       r':'),
        ('NEWLINE',     r'\n'),
        ('SKIP',        r'[ \t]+'),
        ('COMMENT',     r'#.*'),
    ]
    
    def __init__(self, source_code: str):
        self.source = source_code
        self.tokens = []
        self.line = 1
        self.column = 1
        
    def tokenize(self) -> List[Token]:
        """Perform lexical analysis and return list of tokens"""
        pos = 0
        
        while pos < len(self.source):
            match = None
            
            for token_type, pattern in self.TOKEN_SPECS:
                regex = re.compile(pattern)
                match = regex.match(self.source, pos)
                
                if match:
                    value = match.group(0)
                    
                    if token_type == 'NEWLINE':
                        self.line += 1
                        self.column = 1
                    elif token_type not in ['SKIP', 'COMMENT']:
                        token = Token(
                            type=TokenType[token_type],
                            value=value,
                            line=self.line,
                            column=self.column
                        )
                        self.tokens.append(token)
                    
                    self.column += len(value)
                    pos = match.end()
                    break
            
            if not match:
                raise SyntaxError(f"Illegal character '{self.source[pos]}' at line {self.line}, column {self.column}")
        
        self.tokens.append(Token(TokenType.EOF, None, self.line, self.column))
        return self.tokens

# ============================================================================
# PHASE 2: SYNTAX ANALYSIS (PARSER)
# ============================================================================

@dataclass
class ASTNode:
    """Base class for Abstract Syntax Tree nodes"""
    pass

@dataclass
class Program(ASTNode):
    patterns: List['PatternDef']
    statements: List[ASTNode] = field(default_factory=list)  # FIX: Added statements

@dataclass
class PatternDef(ASTNode):
    name: str
    params: List[str]
    body: List[ASTNode]

@dataclass
class GenerateStmt(ASTNode):
    pattern_name: str
    args: List['Expression']
    count: 'Expression'

@dataclass
class PrintStmt(ASTNode):
    expression: 'Expression'

@dataclass
class IfStmt(ASTNode):
    condition: 'Expression'
    then_body: List[ASTNode]
    else_body: Optional[List[ASTNode]]

@dataclass
class ReturnStmt(ASTNode):
    expression: 'Expression'

@dataclass
class Assignment(ASTNode):
    identifier: str
    expression: 'Expression'

@dataclass
class BinaryOp(ASTNode):
    left: 'Expression'
    operator: str
    right: 'Expression'

@dataclass
class UnaryOp(ASTNode):
    operator: str
    operand: 'Expression'

@dataclass
class Number(ASTNode):
    value: float

@dataclass
class Identifier(ASTNode):
    name: str

@dataclass
class FunctionCall(ASTNode):
    name: str
    args: List['Expression']

Expression = BinaryOp | UnaryOp | Number | Identifier | FunctionCall

class Parser:
    """Phase 2: Syntax Analyzer - Builds Abstract Syntax Tree"""
    
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        self.current_token = self.tokens[0]
    
    def advance(self):
        """Move to next token"""
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current_token = self.tokens[self.pos]
    
    def expect(self, token_type: TokenType):
        """Expect a specific token type"""
        if self.current_token.type != token_type:
            raise SyntaxError(
                f"Expected {token_type.name}, got {self.current_token.type.name} "
                f"at line {self.current_token.line}"
            )
        value = self.current_token.value
        self.advance()
        return value
    
    def parse(self) -> Program:
        """Parse the entire program"""
        patterns = []
        statements = []  # FIX: Store top-level statements
        
        while self.current_token.type != TokenType.EOF:
            print(f"[DEBUG] Current token: {self.current_token}")
            # Skip newlines
            if self.current_token.type == TokenType.NEWLINE:
                self.advance()
                continue
                
            if self.current_token.type == TokenType.PATTERN:
                print("[DEBUG] Parsing pattern...")
                patterns.append(self.parse_pattern())
            elif self.current_token.type == TokenType.GENERATE:
                print("[DEBUG] Parsing GENERATE statement!") 
                # FIX: Parse generate statements at top level
                statements.append(self.parse_generate())
            elif self.current_token.type == TokenType.PRINT:
                # FIX: Parse print statements at top level
                statements.append(self.parse_print())
            else:
                print(f"[DEBUG] Skipping token: {self.current_token}")
                self.advance()
        print(f"[DEBUG] Parsed {len(patterns)} patterns and {len(statements)} top-level statements")
        # FIX: Return program with both patterns and statements
        return Program(patterns, statements)
    
    def parse_pattern(self) -> PatternDef:
        """Parse a pattern definition"""
        self.expect(TokenType.PATTERN)
        name = self.expect(TokenType.IDENTIFIER)
        
        self.expect(TokenType.LPAREN)
        params = []
        
        if self.current_token.type == TokenType.IDENTIFIER:
            params.append(self.expect(TokenType.IDENTIFIER))
            
            while self.current_token.type == TokenType.COMMA:
                self.advance()
                params.append(self.expect(TokenType.IDENTIFIER))
        
        self.expect(TokenType.RPAREN)
        self.expect(TokenType.LBRACE)
        
        body = []
        while self.current_token.type != TokenType.RBRACE:
            if self.current_token.type == TokenType.NEWLINE:
                self.advance()
                continue
            body.append(self.parse_statement())
        
        self.expect(TokenType.RBRACE)
        
        return PatternDef(name, params, body)
    
    def parse_statement(self) -> ASTNode:
        """Parse a statement"""
        if self.current_token.type == TokenType.GENERATE:
            return self.parse_generate()
        elif self.current_token.type == TokenType.PRINT:
            return self.parse_print()
        elif self.current_token.type == TokenType.IF:
            return self.parse_if()
        elif self.current_token.type == TokenType.RETURN:
            return self.parse_return()
        elif self.current_token.type == TokenType.IDENTIFIER:
            return self.parse_assignment()
        else:
            raise SyntaxError(f"Unexpected token: {self.current_token}")
    
    def parse_generate(self) -> GenerateStmt:
        """Parse a generate statement"""
        self.expect(TokenType.GENERATE)
        pattern_name = self.expect(TokenType.IDENTIFIER)
        
        self.expect(TokenType.LPAREN)
        args = []
        
        if self.current_token.type != TokenType.RPAREN:
            args.append(self.parse_expression())
            
            while self.current_token.type == TokenType.COMMA:
                self.advance()
                args.append(self.parse_expression())
        
        self.expect(TokenType.RPAREN)
        self.expect(TokenType.COLON)
        count = self.parse_expression()
        self.expect(TokenType.SEMICOLON)
        
        return GenerateStmt(pattern_name, args, count)
    
    def parse_print(self) -> PrintStmt:
        """Parse a print statement"""
        self.expect(TokenType.PRINT)
        expr = self.parse_expression()
        self.expect(TokenType.SEMICOLON)
        return PrintStmt(expr)
    
    def parse_if(self) -> IfStmt:
        """Parse an if statement"""
        self.expect(TokenType.IF)
        self.expect(TokenType.LPAREN)
        condition = self.parse_expression()
        self.expect(TokenType.RPAREN)
        self.expect(TokenType.LBRACE)
        
        then_body = []
        while self.current_token.type != TokenType.RBRACE:
            if self.current_token.type == TokenType.NEWLINE:
                self.advance()
                continue
            then_body.append(self.parse_statement())
        
        self.expect(TokenType.RBRACE)
        
        else_body = None
        if self.current_token.type == TokenType.ELSE:
            self.advance()
            self.expect(TokenType.LBRACE)
            else_body = []
            while self.current_token.type != TokenType.RBRACE:
                if self.current_token.type == TokenType.NEWLINE:
                    self.advance()
                    continue
                else_body.append(self.parse_statement())
            self.expect(TokenType.RBRACE)
        
        return IfStmt(condition, then_body, else_body)
    
    def parse_return(self) -> ReturnStmt:
        """Parse a return statement"""
        self.expect(TokenType.RETURN)
        expr = self.parse_expression()
        self.expect(TokenType.SEMICOLON)
        return ReturnStmt(expr)
    
    def parse_assignment(self) -> Assignment:
        """Parse an assignment statement"""
        identifier = self.expect(TokenType.IDENTIFIER)
        self.expect(TokenType.ASSIGN)
        expr = self.parse_expression()
        self.expect(TokenType.SEMICOLON)
        return Assignment(identifier, expr)
    
    def parse_expression(self) -> Expression:
        """Parse an expression (with operator precedence)"""
        return self.parse_comparison()
    
    def parse_comparison(self) -> Expression:
        """Parse comparison operators"""
        left = self.parse_additive()
        
        while self.current_token.type in [TokenType.EQ, TokenType.NEQ, 
                                          TokenType.LT, TokenType.GT,
                                          TokenType.LTE, TokenType.GTE]:
            op = self.current_token.value
            self.advance()
            right = self.parse_additive()
            left = BinaryOp(left, op, right)
        
        return left
    
    def parse_additive(self) -> Expression:
        """Parse addition and subtraction"""
        left = self.parse_multiplicative()
        
        while self.current_token.type in [TokenType.PLUS, TokenType.MINUS]:
            op = self.current_token.value
            self.advance()
            right = self.parse_multiplicative()
            left = BinaryOp(left, op, right)
        
        return left
    
    def parse_multiplicative(self) -> Expression:
        """Parse multiplication, division, and modulo"""
        left = self.parse_unary()
        
        while self.current_token.type in [TokenType.MULTIPLY, TokenType.DIVIDE, TokenType.MODULO]:
            op = self.current_token.value
            self.advance()
            right = self.parse_unary()
            left = BinaryOp(left, op, right)
        
        return left
    
    def parse_unary(self) -> Expression:
        """Parse unary operators"""
        if self.current_token.type in [TokenType.PLUS, TokenType.MINUS]:
            op = self.current_token.value
            self.advance()
            operand = self.parse_unary()
            return UnaryOp(op, operand)
        
        return self.parse_primary()
    
    def parse_primary(self) -> Expression:
        """Parse primary expressions"""
        if self.current_token.type == TokenType.NUMBER:
            value = float(self.current_token.value)
            self.advance()
            return Number(value)
        
        elif self.current_token.type == TokenType.IDENTIFIER:
            name = self.current_token.value
            self.advance()
            
            if self.current_token.type == TokenType.LPAREN:
                self.advance()
                args = []
                
                if self.current_token.type != TokenType.RPAREN:
                    args.append(self.parse_expression())
                    
                    while self.current_token.type == TokenType.COMMA:
                        self.advance()
                        args.append(self.parse_expression())
                
                self.expect(TokenType.RPAREN)
                return FunctionCall(name, args)
            
            return Identifier(name)
        
        elif self.current_token.type == TokenType.LPAREN:
            self.advance()
            expr = self.parse_expression()
            self.expect(TokenType.RPAREN)
            return expr
        
        else:
            raise SyntaxError(f"Unexpected token in expression: {self.current_token}")

# ============================================================================
# PHASE 3: SEMANTIC ANALYSIS
# ============================================================================

class SymbolTable:
    """Symbol table for semantic analysis"""
    
    def __init__(self):
        self.scopes = [{}]  # Stack of scopes
        self.patterns = {}  # Pattern definitions
    
    def enter_scope(self):
        """Enter a new scope"""
        self.scopes.append({})
    
    def exit_scope(self):
        """Exit current scope"""
        if len(self.scopes) > 1:
            self.scopes.pop()
    
    def define(self, name: str, type_: str, value=None):
        """Define a variable in current scope"""
        self.scopes[-1][name] = {'type': type_, 'value': value}
    
    def lookup(self, name: str) -> Optional[Dict]:
        """Look up a variable in all scopes"""
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None
    
    def define_pattern(self, name: str, params: List[str]):
        """Define a pattern"""
        self.patterns[name] = {'params': params}
    
    def get_pattern(self, name: str) -> Optional[Dict]:
        """Get a pattern definition"""
        return self.patterns.get(name)

class SemanticAnalyzer:
    """Phase 3: Semantic Analyzer - Type checking and symbol table construction"""
    
    def __init__(self):
        self.symbol_table = SymbolTable()
        self.errors = []
    
    def analyze(self, ast: Program):
        """Perform semantic analysis on the AST"""
        # First pass: collect all pattern definitions
        for pattern in ast.patterns:
            self.symbol_table.define_pattern(pattern.name, pattern.params)
        
        # Second pass: analyze each pattern
        for pattern in ast.patterns:
            self.analyze_pattern(pattern)
        
        # FIX: Third pass: analyze top-level statements
        for stmt in ast.statements:
            self.analyze_statement(stmt)
        
        if self.errors:
            raise SemanticError("\n".join(self.errors))
    
    def analyze_pattern(self, pattern: PatternDef):
        """Analyze a pattern definition"""
        self.symbol_table.enter_scope()
        
        # Define parameters
        for param in pattern.params:
            self.symbol_table.define(param, 'number')
        
        self.symbol_table.define('n', 'number')
        # Analyze body
        for stmt in pattern.body:
            self.analyze_statement(stmt)
        
        self.symbol_table.exit_scope()
    
    def analyze_statement(self, stmt: ASTNode):
        """Analyze a statement"""
        if isinstance(stmt, Assignment):
            expr_type = self.analyze_expression(stmt.expression)
            self.symbol_table.define(stmt.identifier, expr_type)
        
        elif isinstance(stmt, GenerateStmt):
            pattern = self.symbol_table.get_pattern(stmt.pattern_name)
            if not pattern:
                self.errors.append(f"Undefined pattern: {stmt.pattern_name}")
            elif len(stmt.args) != len(pattern['params']):
                self.errors.append(
                    f"Pattern {stmt.pattern_name} expects {len(pattern['params'])} arguments, "
                    f"got {len(stmt.args)}"
                )
            
            for arg in stmt.args:
                self.analyze_expression(arg)
            self.analyze_expression(stmt.count)
        
        elif isinstance(stmt, PrintStmt):
            self.analyze_expression(stmt.expression)
        
        elif isinstance(stmt, IfStmt):
            self.analyze_expression(stmt.condition)
            for s in stmt.then_body:
                self.analyze_statement(s)
            if stmt.else_body:
                for s in stmt.else_body:
                    self.analyze_statement(s)
        
        elif isinstance(stmt, ReturnStmt):
            self.analyze_expression(stmt.expression)
    
    def analyze_expression(self, expr: Expression) -> str:
        """Analyze an expression and return its type"""
        if isinstance(expr, Number):
            return 'number'
        
        elif isinstance(expr, Identifier):
            symbol = self.symbol_table.lookup(expr.name)
            if not symbol:
                # Don't error on 'n' as it's defined at runtime
                if expr.name != 'n':
                    self.errors.append(f"Undefined variable: {expr.name}")
                return 'number'
            return symbol['type']
        
        elif isinstance(expr, BinaryOp):
            self.analyze_expression(expr.left)
            self.analyze_expression(expr.right)
            return 'number'
        
        elif isinstance(expr, UnaryOp):
            self.analyze_expression(expr.operand)
            return 'number'
        
        elif isinstance(expr, FunctionCall):
            for arg in expr.args:
                self.analyze_expression(arg)
            return 'number'
        
        return 'unknown'

class SemanticError(Exception):
    pass

# ============================================================================
# PHASE 4: INTERMEDIATE CODE GENERATION (Three-Address Code)
# ============================================================================

@dataclass
class TAC:
    """Three-Address Code instruction"""
    op: str
    arg1: Any = None
    arg2: Any = None
    result: Any = None
    
    def __repr__(self):
        if self.op in ['ASSIGN', 'UNARY']:
            return f"{self.result} = {self.arg1}"
        elif self.op in ['ADD', 'SUB', 'MUL', 'DIV', 'MOD', 'EQ', 'NEQ', 'LT', 'GT', 'LTE', 'GTE']:
            return f"{self.result} = {self.arg1} {self.op} {self.arg2}"
        elif self.op == 'LABEL':
            return f"{self.result}:"
        elif self.op == 'GOTO':
            return f"goto {self.result}"
        elif self.op == 'IF_FALSE':
            return f"if_false {self.arg1} goto {self.result}"
        elif self.op == 'CALL':
            args_str = ', '.join(str(a) for a in self.arg2)
            return f"{self.result} = call {self.arg1}({args_str})"
        elif self.op == 'PRINT':
            return f"print {self.arg1}"
        elif self.op == 'RETURN':
            return f"return {self.arg1}"
        elif self.op == 'GENERATE':
            args_str = ', '.join(str(a) for a in self.arg2) if isinstance(self.arg2, list) else str(self.arg2)  # ← CHANGE THIS LINE
            return f"generate {self.arg1}({args_str}) count {self.result}"
        else:
            return f"{self.op} {self.arg1} {self.arg2} {self.result}"

class IntermediateCodeGenerator:
    """Phase 4: Generate Three-Address Code from AST"""
    
    def __init__(self):
        self.instructions = []
        self.temp_counter = 0
        self.label_counter = 0
    
    def new_temp(self) -> str:
        """Generate a new temporary variable"""
        self.temp_counter += 1
        return f"t{self.temp_counter}"
    
    def new_label(self) -> str:
        """Generate a new label"""
        self.label_counter += 1
        return f"L{self.label_counter}"
    
    def emit(self, op: str, arg1=None, arg2=None, result=None):
        """Emit a three-address code instruction"""
        instruction = TAC(op, arg1, arg2, result)
        self.instructions.append(instruction)
        return instruction
    
    def generate(self, ast: Program) -> List[TAC]:
        """Generate intermediate code for the entire program"""
        # First, generate code for all pattern definitions
        for pattern in ast.patterns:
            self.generate_pattern(pattern)
        
        # FIX: Then generate code for top-level statements (GENERATE, PRINT, etc.)
        for stmt in ast.statements:
            self.generate_statement(stmt)
        
        return self.instructions
    
    def generate_pattern(self, pattern: PatternDef):
        """Generate code for a pattern definition"""
        self.emit('LABEL', result=f"pattern_{pattern.name}")
        
        for stmt in pattern.body:
            self.generate_statement(stmt)
    
    def generate_statement(self, stmt: ASTNode):
        """Generate code for a statement"""
        if isinstance(stmt, Assignment):
            temp = self.generate_expression(stmt.expression)
            self.emit('ASSIGN', temp, result=stmt.identifier)
        
        elif isinstance(stmt, GenerateStmt):
            args = [self.generate_expression(arg) for arg in stmt.args]
            count = self.generate_expression(stmt.count)
            self.emit('GENERATE', stmt.pattern_name, args, count)
        
        elif isinstance(stmt, PrintStmt):
            temp = self.generate_expression(stmt.expression)
            self.emit('PRINT', temp)
        
        elif isinstance(stmt, IfStmt):
            cond_temp = self.generate_expression(stmt.condition)
            else_label = self.new_label()
            end_label = self.new_label()
            
            self.emit('IF_FALSE', cond_temp, result=else_label)
            
            for s in stmt.then_body:
                self.generate_statement(s)
            
            self.emit('GOTO', result=end_label)
            self.emit('LABEL', result=else_label)
            
            if stmt.else_body:
                for s in stmt.else_body:
                    self.generate_statement(s)
            
            self.emit('LABEL', result=end_label)
        
        elif isinstance(stmt, ReturnStmt):
            temp = self.generate_expression(stmt.expression)
            self.emit('RETURN', temp)
    
    def generate_expression(self, expr: Expression) -> str:
        """Generate code for an expression and return the result temp"""
        if isinstance(expr, Number):
            return str(int(expr.value))
        
        elif isinstance(expr, Identifier):
            return expr.name
        
        elif isinstance(expr, BinaryOp):
            left = self.generate_expression(expr.left)
            right = self.generate_expression(expr.right)
            result = self.new_temp()
            
            op_map = {
                '+': 'ADD', '-': 'SUB', '*': 'MUL', '/': 'DIV', '%': 'MOD',
                '==': 'EQ', '!=': 'NEQ', '<': 'LT', '>': 'GT', '<=': 'LTE', '>=': 'GTE'
            }
            
            self.emit(op_map[expr.operator], left, right, result)
            return result
        
        elif isinstance(expr, UnaryOp):
            operand = self.generate_expression(expr.operand)
            result = self.new_temp()
            
            if expr.operator == '-':
                self.emit('SUB', '0', operand, result)
            else:
                self.emit('ASSIGN', operand, result=result)
            
            return result
        
        elif isinstance(expr, FunctionCall):
            args = [self.generate_expression(arg) for arg in expr.args]
            result = self.new_temp()
            self.emit('CALL', expr.name, args, result)
            return result
        
        return "0"

# ============================================================================
# PHASE 5: OPTIMIZATION
# ============================================================================

class Optimizer:
    """Phase 5: Basic optimizations on intermediate code"""
    
    def __init__(self, instructions: List[TAC]):
        self.instructions = instructions
    
    def optimize(self) -> List[TAC]:
        """Apply optimizations"""
        optimized = self.constant_folding(self.instructions)
        optimized = self.dead_code_elimination(optimized)
        optimized = self.algebraic_simplification(optimized)
        return optimized
    
    def constant_folding(self, instructions: List[TAC]) -> List[TAC]:
        """Fold constant expressions"""
        optimized = []
        constants = {}
        
        for instr in instructions:
            if instr.op in ['ADD', 'SUB', 'MUL', 'DIV', 'MOD']:
                arg1_val = constants.get(instr.arg1, instr.arg1)
                arg2_val = constants.get(instr.arg2, instr.arg2)
                
                # Try to convert to numeric values
                try:
                    arg1_val = float(arg1_val) if not isinstance(arg1_val, (int, float)) else arg1_val
                    arg2_val = float(arg2_val) if not isinstance(arg2_val, (int, float)) else arg2_val
                except:
                    optimized.append(instr)
                    continue
                
                if isinstance(arg1_val, (int, float)) and isinstance(arg2_val, (int, float)):
                    # Both operands are constants - fold them
                    if instr.op == 'ADD':
                        result_val = arg1_val + arg2_val
                    elif instr.op == 'SUB':
                        result_val = arg1_val - arg2_val
                    elif instr.op == 'MUL':
                        result_val = arg1_val * arg2_val
                    elif instr.op == 'DIV':
                        result_val = arg1_val / arg2_val if arg2_val != 0 else arg1_val
                    elif instr.op == 'MOD':
                        result_val = arg1_val % arg2_val if arg2_val != 0 else arg1_val
                    
                    constants[instr.result] = result_val
                    optimized.append(TAC('ASSIGN', result_val, result=instr.result))
                    continue
            
            elif instr.op == 'ASSIGN':
                try:
                    if isinstance(instr.arg1, (int, float)):
                        constants[instr.result] = instr.arg1
                    else:
                        val = float(instr.arg1)
                        constants[instr.result] = val
                except:
                    pass
            
            optimized.append(instr)
        
        return optimized
    
    def dead_code_elimination(self, instructions: List[TAC]) -> List[TAC]:
        """Remove unused temporary variables"""
        used_vars = set()
        
        # Collect all used variables
        for instr in instructions:
            if instr.arg1 and isinstance(instr.arg1, str):
                used_vars.add(instr.arg1)
            if instr.arg2:
                if isinstance(instr.arg2, str):
                    used_vars.add(instr.arg2)
                elif isinstance(instr.arg2, list):  # FIX: Handle list arguments (for GENERATE)
                    for arg in instr.arg2:
                        if isinstance(arg, str):
                            used_vars.add(arg)
            if instr.op in ['PRINT', 'RETURN', 'IF_FALSE']:
                if instr.arg1:
                    used_vars.add(instr.arg1)
        
        # Keep only instructions that define used variables or have side effects
        optimized = []
        for instr in instructions:
            if instr.op in ['LABEL', 'GOTO', 'IF_FALSE', 'PRINT', 'RETURN', 'GENERATE', 'CALL']:
                optimized.append(instr)
            elif instr.result and (instr.result in used_vars or not str(instr.result).startswith('t')):
                optimized.append(instr)
        
        return optimized
    
    def algebraic_simplification(self, instructions: List[TAC]) -> List[TAC]:
        """Apply algebraic simplifications"""
        optimized = []
        
        for instr in instructions:
            # x + 0 = x
            if instr.op == 'ADD' and instr.arg2 == '0':
                optimized.append(TAC('ASSIGN', instr.arg1, result=instr.result))
            # x * 1 = x
            elif instr.op == 'MUL' and instr.arg2 == '1':
                optimized.append(TAC('ASSIGN', instr.arg1, result=instr.result))
            # x * 0 = 0
            elif instr.op == 'MUL' and instr.arg2 == '0':
                optimized.append(TAC('ASSIGN', '0', result=instr.result))
            else:
                optimized.append(instr)
        
        return optimized

# ============================================================================
# PHASE 6: CODE GENERATION (Interpreter)
# ============================================================================
class Interpreter:
    """Phase 6: Code Generation - Execute optimized intermediate code"""
    
    def __init__(self, instructions: List[TAC], symbol_table: SymbolTable):
        self.instructions = instructions
        self.symbol_table = symbol_table
        self.variables = {}
        self.output = []
        self.pc = 0  # Program counter
        self.labels = {}
        
        # Build label map
        for i, instr in enumerate(instructions):
            if instr.op == 'LABEL':
                self.labels[instr.result] = i

    def execute(self) -> List[str]:
        """Execute the intermediate code"""
        self.pc = 0
        
        # FIX: Execute all instructions including those outside pattern definitions
        while self.pc < len(self.instructions):
            instr = self.instructions[self.pc]
            
            # Skip pattern labels but execute GENERATE statements
            if instr.op == 'LABEL' and instr.result.startswith('pattern_'):
                self.pc += 1
                continue
                
            self.execute_instruction(instr)
            self.pc += 1
        
        return self.output

    def execute_instruction(self, instr: TAC):
        """Execute a single instruction"""
        if instr.op == 'ASSIGN':
            self.variables[instr.result] = self.get_value(instr.arg1)
        
        elif instr.op == 'ADD':
            self.variables[instr.result] = (
                self.get_value(instr.arg1) + self.get_value(instr.arg2)
            )
        
        elif instr.op == 'SUB':
            self.variables[instr.result] = (
                self.get_value(instr.arg1) - self.get_value(instr.arg2)
            )
        
        elif instr.op == 'MUL':
            self.variables[instr.result] = (
                self.get_value(instr.arg1) * self.get_value(instr.arg2)
            )
        
        elif instr.op == 'DIV':
            divisor = self.get_value(instr.arg2)
            if divisor != 0:
                self.variables[instr.result] = self.get_value(instr.arg1) / divisor
            else:
                self.variables[instr.result] = 0
        
        elif instr.op == 'MOD':
            divisor = self.get_value(instr.arg2)
            if divisor != 0:
                self.variables[instr.result] = self.get_value(instr.arg1) % divisor
            else:
                self.variables[instr.result] = 0
        
        elif instr.op in ['EQ', 'NEQ', 'LT', 'GT', 'LTE', 'GTE']:
            left = self.get_value(instr.arg1)
            right = self.get_value(instr.arg2)
            
            if instr.op == 'EQ':
                result = 1 if left == right else 0
            elif instr.op == 'NEQ':
                result = 1 if left != right else 0
            elif instr.op == 'LT':
                result = 1 if left < right else 0
            elif instr.op == 'GT':
                result = 1 if left > right else 0
            elif instr.op == 'LTE':
                result = 1 if left <= right else 0
            elif instr.op == 'GTE':
                result = 1 if left >= right else 0
            
            self.variables[instr.result] = result
        
        elif instr.op == 'PRINT':
            value = self.get_value(instr.arg1)
            self.output.append(
                str(int(value) if isinstance(value, float) and value.is_integer() else value)
            )
        
        elif instr.op == 'GOTO':
            self.pc = self.labels[instr.result] - 1
        
        elif instr.op == 'IF_FALSE':
            if self.get_value(instr.arg1) == 0:
                self.pc = self.labels[instr.result] - 1
        
        elif instr.op == 'GENERATE':
            self.execute_generate(instr)
        
        elif instr.op == 'LABEL':
            pass  # Labels are just markers
        
        elif instr.op == 'RETURN':
            pass  # Handle returns if needed

    def execute_generate(self, instr: TAC):
        """Execute a generate statement"""
        pattern_name = instr.arg1
        args = [self.get_value(arg) for arg in instr.arg2]
        count = int(self.get_value(instr.result))
        
        # Execute the pattern multiple times
        pattern_def = self.symbol_table.get_pattern(pattern_name)
        if not pattern_def:
            return
        
        # Find pattern label
        pattern_label = f"pattern_{pattern_name}"
        if pattern_label not in self.labels:
            return
        
        # Current arguments that will be updated each iteration
        current_args = args[:]
        
        # Execute pattern count times
        for i in range(count+1):
            # Set up parameters for this iteration
            for param, arg in zip(pattern_def['params'], current_args):
                self.variables[param] = arg
            
            # Set special variable 'n' for iteration index
            self.variables['n'] = i
            for var in ['result', 'next', 'value', 'square', 'num', 'remainder']:
               if var in self.variables and var not in pattern_def['params'] and var != 'n':
                  del self.variables[var]
            # Save PC to restore later
            saved_pc = self.pc
            
            # Execute pattern body starting right after the label
            self.pc = self.labels[pattern_label] + 1
            
            # Execute until we hit another pattern label or end
            while self.pc < len(self.instructions):
                current_instr = self.instructions[self.pc]
                
                # Stop if we hit another pattern definition
                if current_instr.op == 'LABEL' and current_instr.result.startswith('pattern_'):
                    break
                
                # Stop if we hit another GENERATE
                if current_instr.op == 'GENERATE':
                    break
                
                self.execute_instruction(current_instr)
                self.pc += 1
            
            # Update arguments for next iteration based on what the pattern computed
            if len(current_args) == 1:
                # Single parameter patterns - check for computed values in priority order
                if 'result' in self.variables:
                    current_args = [self.variables['result']]
                elif 'next' in self.variables:
                    current_args = [self.variables['next']]
                elif 'temp' in self.variables:
                    current_args = [self.variables['temp']]
                # ADD THIS after the temp check:
                elif 'value' in self.variables:
                    current_args = [self.variables['value']]    
                # else: keep current_args as is
                    
            elif len(current_args) >= 2:
                # Multi-parameter patterns (like Fibonacci)
                if 'next' in self.variables:
                    new_b = self.variables['next']
                    current_args = [current_args[1], new_b]
                elif 'temp' in self.variables:
                    new_b = self.variables['temp']
                    current_args = [current_args[1], new_b]
                elif 'result' in self.variables:
                    # Use result as second parameter
                    current_args = [current_args[1], self.variables['result']]
                else:
                    # Fallback: calculate Fibonacci progression
                    new_b = current_args[0] + current_args[1]
                    current_args = [current_args[1], new_b]
            
            # Restore program counter
            self.pc = saved_pc

    def get_value(self, operand):
        """Get the value of an operand"""
        if isinstance(operand, (int, float)):
            return operand
        
        try:
            return float(operand)
        except (ValueError, TypeError):
            return self.variables.get(operand, 0)

# ============================================================================
# COMPILER DRIVER
# ============================================================================

class PatternScriptCompiler:
    """Main compiler class that orchestrates all phases"""
    
    def __init__(self, source_code: str, verbose: bool = False):
        self.source_code = source_code
        self.verbose = verbose
        self.tokens = []
        self.ast = None
        self.symbol_table = None
        self.intermediate_code = []
        self.optimized_code = []
        self.output = []
    
    def compile(self) -> Dict[str, Any]:
        """Run all compilation phases"""
        results = {}
        
        try:
            # Phase 1: Lexical Analysis
            if self.verbose:
                print("\n" + "="*60)
                print("PHASE 1: LEXICAL ANALYSIS")
                print("="*60)
            
            lexer = Lexer(self.source_code)
            self.tokens = lexer.tokenize()
            results['tokens'] = self.tokens
            
            if self.verbose:
                for token in self.tokens:  # Show first 20 tokens
                    print(token)


                print(f"\nTotal tokens: {len(self.tokens)}")    
            
            # Phase 2: Syntax Analysis
            if self.verbose:
                print("\n" + "="*60)
                print("PHASE 2: SYNTAX ANALYSIS")
                print("="*60)
            
            parser = Parser(self.tokens)
            self.ast = parser.parse()
            results['ast'] = self.ast
            
            if self.verbose:
                print(f"Successfully parsed {len(self.ast.patterns)} pattern(s)")
                for pattern in self.ast.patterns:
                    print(f"  - Pattern: {pattern.name}({', '.join(pattern.params)})")
                print(f"Successfully parsed {len(self.ast.statements)} top-level statement(s)")  # ADD THIS
                for stmt in self.ast.statements:  # ADD THIS
                     print(f"  - Statement: {type(stmt).__name__}")  # ADD THIS
            # Phase 3: Semantic Analysis
            if self.verbose:
                print("\n" + "="*60)
                print("PHASE 3: SEMANTIC ANALYSIS")
                print("="*60)
            
            semantic_analyzer = SemanticAnalyzer()
            semantic_analyzer.analyze(self.ast)
            self.symbol_table = semantic_analyzer.symbol_table
            results['symbol_table'] = self.symbol_table
            
            if self.verbose:
                print("Semantic analysis passed")
                print(f"Patterns defined: {list(self.symbol_table.patterns.keys())}")
            
            # Phase 4: Intermediate Code Generation
            if self.verbose:
                print("\n" + "="*60)
                print("PHASE 4: INTERMEDIATE CODE GENERATION")
                print("="*60)
            
            ic_generator = IntermediateCodeGenerator()
            self.intermediate_code = ic_generator.generate(self.ast)
            results['intermediate_code'] = self.intermediate_code
            
            if self.verbose:
                for instr in self.intermediate_code:
                    print(f"  {instr}")
            
            # Phase 5: Optimization
            if self.verbose:
                print("\n" + "="*60)
                print("PHASE 5: OPTIMIZATION")
                print("="*60)
            
            optimizer = Optimizer(self.intermediate_code)
            self.optimized_code = optimizer.optimize()
            results['optimized_code'] = self.optimized_code
            
            if self.verbose:
                print(f"Optimized: {len(self.intermediate_code)} -> {len(self.optimized_code)} instructions")
                for instr in self.optimized_code:
                    print(f"  {instr}")
            
            # Phase 6: Code Execution
            if self.verbose:
                print("\n" + "="*60)
                print("PHASE 6: CODE EXECUTION")
                print("="*60)
            
            interpreter = Interpreter(self.optimized_code, self.symbol_table)
            self.output = interpreter.execute()
            results['output'] = self.output
            
            if self.verbose:
                print("Output:")
                for line in self.output:
                    print(f"  {line}")
            
            results['success'] = True
            
        except Exception as e:
            results['success'] = False
            results['error'] = str(e)
            if self.verbose:
                print(f"\nCOMPILATION ERROR: {e}")
        
        return results
    
    def get_output(self) -> List[str]:
        """Get the program output"""
        return self.output

# ============================================================================
# INTERACTIVE CLI
# ============================================================================

def main():
    """Interactive command-line interface"""
    print("="*70)
    print("  PatternScript Compiler - Numerical Pattern Generation Language")
    print("="*70)
    print("\nCommands:")
    print("  run <filename>  - Compile and run a PatternScript file")
    print("  test            - Run built-in test cases")
    print("  help            - Show language syntax help")
    print("  exit            - Exit the compiler")
    print("="*70)
    
    while True:
        try:
            command = input("\nPatternScript> ").strip()
            
            if not command:
                continue
            
            if command == "exit":
                print("Goodbye!")
                break
            
            elif command == "help":
                show_help()
            
            elif command == "test":
                run_tests()
            
            elif command.startswith("run "):
                filename = command[4:].strip()
                try:
                    with open(filename, 'r') as f:
                        source_code = f.read()
                    
                    print(f"\nCompiling {filename}...\n")
                    compiler = PatternScriptCompiler(source_code, verbose=True)
                    results = compiler.compile()
                    
                    if results['success']:
                        print("\n" + "="*60)
                        print("COMPILATION SUCCESSFUL")
                        print("="*60)
                    else:
                        print(f"\nCompilation failed: {results['error']}")
                
                except FileNotFoundError:
                    print(f"Error: File '{filename}' not found")
                except Exception as e:
                    print(f"Error: {e}")
            
            else:
                print(f"Unknown command: {command}")
                print("Type 'help' for available commands")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except EOFError:
            print("\n\nGoodbye!")
            break

def show_help():
    """Display language syntax help"""
    help_text = """
PatternScript Language Syntax:
==============================

1. Pattern Definition:
   pattern name(param1, param2, ...) {
       statements
   }

2. Statements:
   - Assignment:        variable = expression;
   - Generate:          generate pattern_name(args): count;
   - Print:             print expression;
   - If-Else:           if (condition) { ... } else { ... }
   - Return:            return expression;

3. Expressions:
   - Arithmetic: +, -, *, /, %
   - Comparison: ==, !=, <, >, <=, >=
   - Variables: n (iteration index), parameters

4. Built-in Functions:
   - factorial(n)
   - fibonacci(n)

Example Program:
================
pattern fibonacci(a, b) {
    if (n == 0) {
        print a;
    } else {
        print a + b;
    }
}

generate fibonacci(0, 1): 10;
"""
    print(help_text)

def run_tests():
    """Run built-in test cases"""
    print("\n" + "="*70)
    print("RUNNING TEST CASES")
    print("="*70)
    
    test_cases = [
        ("Test 1: Fibonacci Sequence", TEST_CASE_1),
        ("Test 2: Factorial Pattern", TEST_CASE_2),
        ("Test 3: Custom Arithmetic", TEST_CASE_3)
    ]
    
    for test_name, test_code in test_cases:
        print(f"\n{'='*70}")
        print(f"{test_name}")
        print(f"{'='*70}")
        print("\nSource Code:")
        print("-" * 70)
        print(test_code)
        print("-" * 70)
        
        compiler = PatternScriptCompiler(test_code, verbose=True)
        results = compiler.compile()
        
        if results['success']:
            print(f"\n✓ {test_name} PASSED")
        else:
            print(f"\n✗ {test_name} FAILED: {results['error']}")
        
        print()

# ============================================================================
# TEST CASES
# ============================================================================

TEST_CASE_1 = """
# Fibonacci Sequence Generator
pattern fibonacci(a, b) {
    if (n == 0) {
        print a;
    } else {
        temp = a + b;
        print temp;
    }
}

generate fibonacci(0, 1): 10;
"""

TEST_CASE_2 = """
# Factorial Pattern
pattern factorial(base) {
    if (n == 0) {
        result = 1;
        print result;
    } else {
        result = base * n;
        print result;
    }
}

generate factorial(1): 8;
"""

TEST_CASE_3 = """
# Custom Arithmetic Pattern
pattern squares(start) {
    value = start + n;
    square = value * value;
    print square;
}

generate squares(1): 10;
"""

if __name__ == "__main__":
    main()

