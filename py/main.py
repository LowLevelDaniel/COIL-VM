#!/usr/bin/env python3
"""
COIL Virtual Machine - A simple interpreter for COIL binary format
Version 0.1.0

This VM interprets COIL Object Format (.cof/.coil) files directly instead of 
compiling them to native code. It is primarily intended for testing compilers
that target COIL version 1.0.0.
"""

import struct
import sys
import os
import argparse
from enum import Enum, IntEnum, auto
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union, BinaryIO

# =============================================================================
# Constants and Enumerations
# =============================================================================

class OpcodeType(IntEnum):
  INSTRUCTION = 0
  DIRECTIVE = 1
  EXTENSION = 2

class Opcode(IntEnum):
  # Control Flow (0x01 - 0x0F)
  NOP   = 0x00  # No operation
  SYMB  = 0x01  # Define a symbol
  BR    = 0x02  # Unconditional branch
  BRC   = 0x03  # Conditional branch
  CALL  = 0x04  # Call subroutine
  RET   = 0x05  # Return from subroutine
  INT   = 0x06  # Trigger interrupt
  IRET  = 0x07  # Return from interrupt
  WFI   = 0x08  # Wait for interrupt
  SYSC  = 0x09  # System call
  WFE   = 0x0A  # Wait for event
  SEV   = 0x0B  # Send event
  TRAP  = 0x0C  # Software trap
  HLT   = 0x0D  # Halt execution
  
  # Arithmetic Operations (0x10 - 0x1F)
  ADD   = 0x10  # Addition
  SUB   = 0x11  # Subtraction
  MUL   = 0x12  # Multiplication
  DIV   = 0x13  # Division
  MOD   = 0x14  # Modulus
  NEG   = 0x15  # Negation
  ABS   = 0x16  # Absolute value
  INC   = 0x17  # Increment
  DEC   = 0x18  # Decrement
  ADDC  = 0x19  # Add with carry
  SUBB  = 0x1A  # Subtract with borrow
  MULH  = 0x1B  # Multiplication high bits
  
  # Bit Manipulation (0x20 - 0x2F)
  AND   = 0x20  # Bitwise AND
  OR    = 0x21  # Bitwise OR
  XOR   = 0x22  # Bitwise XOR
  NOT   = 0x23  # Bitwise NOT
  SHL   = 0x24  # Shift left
  SHR   = 0x25  # Shift right logical
  SAR   = 0x26  # Shift right arithmetic
  ROL   = 0x27  # Rotate left
  ROR   = 0x28  # Rotate right
  CLZ   = 0x29  # Count leading zeros
  CTZ   = 0x2A  # Count trailing zeros
  POPC  = 0x2B  # Population count
  BSWP  = 0x2C  # Byte swap
  BEXT  = 0x2D  # Bit extraction
  BINS  = 0x2E  # Bit insertion
  BMSK  = 0x2F  # Bit mask
  
  # Comparison Operations (0x30 - 0x3F)
  CMP   = 0x30  # Compare values and set flags
  TEST  = 0x31  # Test bits
  
  # Data Movement (0x40 - 0x4F)
  MOV   = 0x40  # Move data
  LOAD  = 0x41  # Load from memory
  STORE = 0x42  # Store to memory
  XCHG  = 0x43  # Exchange data
  LEA   = 0x44  # Load effective address
  MOVI  = 0x45  # Move immediate value
  MOVZ  = 0x46  # Move with zero extension
  MOVS  = 0x47  # Move with sign extension
  LDMUL = 0x48  # Load multiple
  STMUL = 0x49  # Store multiple
  LDSTR = 0x4A  # Load string
  STSTR = 0x4B  # Store string
  
  # Stack Operations (0x50 - 0x5F)
  PUSH  = 0x50  # Push onto stack
  POP   = 0x51  # Pop from stack
  PUSHA = 0x52  # Push all registers
  POPA  = 0x53  # Pop all registers
  PUSHF = 0x54  # Push flags
  POPF  = 0x55  # Pop flags
  ADJSP = 0x56  # Adjust stack pointer
  
  # Variable Operations (0x60 - 0x6F)
  VARCR = 0x60  # Create variable
  VARDL = 0x61  # Delete variable
  VARSC = 0x62  # Create variable scope
  VAREND = 0x63  # End variable scope
  VARGET = 0x64  # Get variable value
  VARSET = 0x65  # Set variable value
  VARREF = 0x66  # Get variable reference
  
  # Conversion Operations (0x70 - 0x7F)
  TRUNC = 0x70  # Truncate value
  ZEXT  = 0x71  # Zero extend value
  SEXT  = 0x72  # Sign extend value
  FTOI  = 0x73  # Float to integer
  ITOF  = 0x74  # Integer to float
  FTOB  = 0x75  # Float to bits
  BTOF  = 0x76  # Bits to float
  F32F64 = 0x77  # Float32 to Float64
  F64F32 = 0x78  # Float64 to Float32
  
  # Directive opcodes (0xD0 - 0xDF)
  DIR_VERSION = 0xD0  # Version specification
  DIR_TARGET  = 0xD1  # Target architecture
  DIR_SECTION = 0xD2  # Section definition
  DIR_SYMBOL  = 0xD3  # Symbol definition
  DIR_ALIGN   = 0xD4  # Alignment directive
  DIR_DATA    = 0xD5  # Data definition
  DIR_ABI     = 0xD6  # ABI definition
  DIR_FEATURE = 0xD7  # Feature control
  DIR_OPTIMIZE = 0xD8  # Optimization control

class OperandQualifier(IntEnum):
  IMM = 0x01  # Immediate value
  VAR = 0x02  # Variable
  REG = 0x03  # Register
  MEM = 0x04  # Memory address
  LBL = 0x05  # Label
  STR = 0x06  # String literal
  SYM = 0x07  # Symbol
  REL = 0x08  # Relative offset

class CoilType(IntEnum):
  INT   = 0x00  # Signed integer
  UINT  = 0x01  # Unsigned integer
  FLOAT = 0x02  # Floating point
  
  VOID  = 0xF0  # Void type
  BOOL  = 0xF1  # Boolean
  LINT  = 0xF2  # Largest native integer
  FINT  = 0xF3  # Fastest native integer
  PTR   = 0xF4  # Pointer type
  
  PARAM2 = 0xFD  # Parameter type 2
  PARAM1 = 0xFE  # Parameter type 1
  PARAM0 = 0xFF  # Parameter type 0

class BranchCondition(IntEnum):
  ALWAYS = 0x00  # Always branch
  EQ     = 0x01  # Equal / Zero
  NE     = 0x02  # Not equal / Not zero
  LT     = 0x03  # Less than
  LE     = 0x04  # Less than or equal
  GT     = 0x05  # Greater than
  GE     = 0x06  # Greater than or equal
  CARRY  = 0x07  # Carry flag set
  OFLOW  = 0x08  # Overflow flag set
  SIGN   = 0x09  # Sign flag set
  PARITY = 0x0A  # Parity flag set

class SectionType(IntEnum):
  NULL      = 0    # Unused section
  CODE      = 1    # Executable code
  DATA      = 2    # Initialized data
  BSS       = 3    # Uninitialized data
  SYMTAB    = 4    # Symbol table
  STRTAB    = 5    # String table
  RELA      = 6    # Relocation entries with addends
  REL       = 7    # Relocation entries without addends
  METADATA  = 8    # Metadata
  COMMENT   = 9    # Comment information
  DIRECTIVE = 10   # Directives for the assembler

class SymbolType(IntEnum):
  NOTYPE  = 0    # No type specified
  OBJECT  = 1    # Data object
  FUNC    = 2    # Function or code
  SECTION = 3    # Section symbol
  FILE    = 4    # Source file name

class SymbolBinding(IntEnum):
  LOCAL  = 0    # Local symbol
  GLOBAL = 1    # Global symbol
  WEAK   = 2    # Weak symbol

# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Operand:
  qualifier: OperandQualifier
  type: CoilType
  type_width: Optional[int] = None  # For types that need width
  value: Optional[any] = None
  
  def __str__(self):
    if self.qualifier == OperandQualifier.REG:
      reg_prefix = "R"
      if self.type_width is not None:
        if self.type_width == 8:
          reg_prefix += "B"
        elif self.type_width == 16:
          reg_prefix += "W"
        elif self.type_width == 32:
          reg_prefix += "L"
        elif self.type_width == 64:
          reg_prefix += "Q"
      return f"{reg_prefix}{self.value}"
    elif self.qualifier == OperandQualifier.IMM:
      return f"{self.value}"
    else:
      return f"{self.qualifier.name}:{self.type.name}:{self.value}"

@dataclass
class Instruction:
  opcode: Opcode
  qualifier: int
  operands: List[Operand]
  
  def __str__(self):
    operand_str = ", ".join(str(op) for op in self.operands)
    return f"{self.opcode.name} {operand_str}"

@dataclass
class Symbol:
  name: str
  value: int
  size: int
  type: SymbolType
  binding: SymbolBinding
  visibility: int
  section_index: int

@dataclass
class Section:
  name: str
  type: SectionType
  flags: int
  data: bytearray
  address: int = 0
  link: int = 0
  info: int = 0
  alignment: int = 1
  entsize: int = 0

@dataclass
class COFHeader:
  magic: int
  version_major: int
  version_minor: int
  version_patch: int
  flags: int
  target: int
  section_count: int
  entrypoint: int
  str_tab_off: int
  str_tab_size: int
  sym_tab_off: int
  sym_tab_size: int

# =============================================================================
# Virtual Machine State
# =============================================================================

class CoilVM:
  def __init__(self):
    # Initialize virtual machine state
    self.memory = bytearray(1024 * 1024)  # 1MB memory space
    self.registers = [0] * 16  # 16 general purpose 64-bit registers
    self.ip = 0  # Instruction pointer
    self.sp = len(self.memory) - 1024  # Stack pointer starts at high memory
    self.bp = self.sp  # Base pointer
    
    # Flags register
    self.flags = {
      'zero': False,  # Result is zero
      'sign': False,  # Result is negative
      'carry': False,  # Operation generated a carry
      'overflow': False,  # Operation generated an overflow
      'parity': False,  # Parity of result
    }
    
    # COIL Object Format data
    self.sections: List[Section] = []
    self.symbols: List[Symbol] = []
    self.string_table: bytes = b''
    self.code_sections: List[Section] = []
    self.current_section_index = 0
    self.current_instruction_index = 0
    
    # Program state
    self.running = False
    self.debug = False
    self.halted = False
    self.big_endian_magic = False  # Flag for big-endian magic number
    
    # Variable storage
    self.variables = {}
    self.scopes = [{}]
    
  def load_file(self, filename: str) -> bool:
    """Load a COIL object file into the VM."""
    try:
      with open(filename, 'rb') as f:
        # Read and parse the COF header
        header = self._parse_cof_header(f)
        print(f"COIL file loaded: v{header.version_major}.{header.version_minor}.{header.version_patch}")
        
        # Parse all section headers
        for i in range(header.section_count):
          section_header = self._parse_section_header(f)
          section_offset = f.tell()
          
          # Save the current file position
          current_pos = f.tell()
          
          # Read section data
          f.seek(section_header.offset)
          section_data = bytearray(f.read(section_header.size))
          
          # Create section object
          section = Section(
            name=self._get_string_from_offset(section_header.name_offset, f),
            type=SectionType(section_header.type),
            flags=section_header.flags,
            data=section_data,
            address=0,  # Will be assigned during loading
            link=section_header.link,
            info=section_header.info,
            alignment=section_header.alignment,
            entsize=section_header.entsize
          )
          
          self.sections.append(section)
          
          # Track code sections separately for execution
          if section.type == SectionType.CODE:
            self.code_sections.append(section)
          
          # Restore file position for reading next section header
          f.seek(current_pos)
        
        # Load string table
        for section in self.sections:
          if section.type == SectionType.STRTAB:
            self.string_table = bytes(section.data)
            break
            
        # Load symbol table
        for section in self.sections:
          if section.type == SectionType.SYMTAB:
            self._parse_symbols(section)
            break
            
        # Load code into memory
        load_addr = 0x1000  # Start at a reasonable offset
        for section in self.sections:
          if section.type in [SectionType.CODE, SectionType.DATA]:
            # Align address as required
            if section.alignment > 1:
              align_mask = section.alignment - 1
              load_addr = (load_addr + align_mask) & ~align_mask
            
            # Copy section data to memory
            section.address = load_addr
            self.memory[load_addr:load_addr + len(section.data)] = section.data
            load_addr += len(section.data)
        
        # Set instruction pointer to entrypoint if provided, otherwise to first code section
        if header.entrypoint != 0:
          self.ip = header.entrypoint
        elif self.code_sections:
          self.ip = self.code_sections[0].address
          
        return True
    except Exception as e:
      print(f"Error loading COIL file: {e}")
      return False
    
  def _parse_cof_header(self, f: BinaryIO) -> COFHeader:
    """Parse the COF header from the file."""
    data = f.read(32)  # Fixed header size is 32 bytes
    
    # First check if magic number is in big-endian format
    be_magic = struct.unpack('>I', data[0:4])[0]
    le_magic = struct.unpack('<I', data[0:4])[0]
    
    if be_magic == 0x434F494C:  # 'COIL' in big-endian
      print("Detected big-endian magic number ('COIL' as 'LIOC'), adjusting interpretation.")
      self.big_endian_magic = True
      magic = be_magic
    else:
      magic = le_magic
    
    # Verify magic number
    if magic != 0x434F494C:
      raise ValueError(f"Invalid COF file: Magic number mismatch. Expected 0x434F494C, got {hex(magic)}")
    
    # Parse the rest of the header fields (still little-endian)
    version_major, version_minor, version_patch, flags, target, section_count = struct.unpack('<BBBBHH', data[4:12])
    entrypoint, str_tab_off, str_tab_size, sym_tab_off, sym_tab_size = struct.unpack('<IIIII', data[12:32])
    
    return COFHeader(
      magic=magic,
      version_major=version_major,
      version_minor=version_minor,
      version_patch=version_patch,
      flags=flags,
      target=target,
      section_count=section_count,
      entrypoint=entrypoint,
      str_tab_off=str_tab_off,
      str_tab_size=str_tab_size,
      sym_tab_off=sym_tab_off,
      sym_tab_size=sym_tab_size
    )
    
  def _parse_section_header(self, f: BinaryIO) -> dict:
    """Parse a section header from the file."""
    data = f.read(36)  # Fixed section header size is 36 bytes
    
    name_offset, type_val, flags, offset, size, link, info, alignment, entsize = \
      struct.unpack('<IIIIIIIII', data)
    
    return {
      'name_offset': name_offset,
      'type': type_val,
      'flags': flags,
      'offset': offset,
      'size': size,
      'link': link,
      'info': info,
      'alignment': alignment,
      'entsize': entsize
    }
    
  def _get_string_from_offset(self, offset: int, f: BinaryIO) -> str:
    """Read a null-terminated string from the string table."""
    if not self.string_table:
      # If string table is not loaded yet, seek to the offset in the file
      current_pos = f.tell()
      f.seek(offset)
      result = ""
      c = f.read(1)
      while c and c != b'\0':
        result += c.decode('utf-8')
        c = f.read(1)
      f.seek(current_pos)
      return result
    else:
      # Extract string from already loaded string table
      end = self.string_table.find(b'\0', offset)
      if end == -1:
        return self.string_table[offset:].decode('utf-8')
      else:
        return self.string_table[offset:end].decode('utf-8')
    
  def _parse_symbols(self, section: Section) -> None:
    """Parse the symbol table from a section."""
    data = section.data
    entry_size = section.entsize if section.entsize > 0 else 16  # Default symbol entry size
    
    offset = 0
    while offset + entry_size <= len(data):
      name_offset, value, size, type_val, binding, visibility, section_index = \
        struct.unpack('<IIIBBBB', data[offset:offset + 16])
      
      symbol = Symbol(
        name=self._get_string_from_table(name_offset),
        value=value,
        size=size,
        type=SymbolType(type_val),
        binding=SymbolBinding(binding),
        visibility=visibility,
        section_index=section_index
      )
      
      self.symbols.append(symbol)
      offset += entry_size
    
  def _get_string_from_table(self, offset: int) -> str:
    """Get a string from the string table."""
    if offset >= len(self.string_table):
      return ""
      
    end = self.string_table.find(b'\0', offset)
    if end == -1:
      return self.string_table[offset:].decode('utf-8')
    else:
      return self.string_table[offset:end].decode('utf-8')
    
  def run(self) -> None:
    """Run the loaded COIL program."""
    if not self.code_sections:
      print("No code sections found in the file.")
      return
      
    self.running = True
    self.current_section_index = 0
    self.current_instruction_index = 0
    
    print(f"Starting execution at IP: {hex(self.ip)}")
    
    try:
      while self.running and not self.halted:
        instruction = self._fetch_decode()
        if instruction:
          if self.debug:
            print(f"Executing: {instruction}")
          self._execute(instruction)
        else:
          # End of code section, check if there are more
          self.current_section_index += 1
          self.current_instruction_index = 0
          
          if self.current_section_index >= len(self.code_sections):
            # No more code sections to execute
            self.running = False
          else:
            # Move to next code section
            self.ip = self.code_sections[self.current_section_index].address
    except Exception as e:
      print(f"Execution error at IP {hex(self.ip)}: {e}")
      
    print("Program execution completed.")
    
  def _fetch_decode(self) -> Optional[Instruction]:
    """Fetch and decode the next instruction."""
    try:
      current_section = self.code_sections[self.current_section_index]
      section_offset = self.ip - current_section.address
      
      if section_offset >= len(current_section.data):
        return None
      
      # Read opcode, qualifier, and operand count
      opcode_val = current_section.data[section_offset]
      qualifier_val = current_section.data[section_offset + 1]
      operand_count = current_section.data[section_offset + 2]
      
      # Determine if this is an instruction or directive
      opcode_type = OpcodeType.INSTRUCTION
      if 0xD0 <= opcode_val <= 0xDF:
        opcode_type = OpcodeType.DIRECTIVE
      elif 0xE0 <= opcode_val <= 0xFF:
        opcode_type = OpcodeType.EXTENSION
      
      # Create instruction object
      instruction = Instruction(
        opcode=Opcode(opcode_val),
        qualifier=qualifier_val,
        operands=[]
      )
      
      # Update instruction pointer
      current_offset = section_offset + 3
      
      # Decode operands
      for i in range(operand_count):
        if current_offset + 2 >= len(current_section.data):
          # Incomplete instruction
          break
          
        # Read operand qualifier and type
        operand_qualifier = current_section.data[current_offset]
        operand_type = current_section.data[current_offset + 1]
        current_offset += 2
        
        # Create operand object
        operand = Operand(
          qualifier=OperandQualifier(operand_qualifier),
          type=CoilType(operand_type)
        )
        
        # Read type width for types that need it
        if operand.type in [CoilType.INT, CoilType.UINT, CoilType.FLOAT]:
          if current_offset < len(current_section.data):
            operand.type_width = current_section.data[current_offset]
            current_offset += 1
        
        # Read operand value based on qualifier
        if operand.qualifier == OperandQualifier.IMM:
          # Immediate value size depends on type
          width = operand.type_width if operand.type_width else 32
          size = width // 8
          
          if current_offset + size <= len(current_section.data):
            if width == 8:
              operand.value = current_section.data[current_offset]
            elif width == 16:
              operand.value = struct.unpack('<H', current_section.data[current_offset:current_offset + 2])[0]
            elif width == 32:
              operand.value = struct.unpack('<I', current_section.data[current_offset:current_offset + 4])[0]
            elif width == 64:
              operand.value = struct.unpack('<Q', current_section.data[current_offset:current_offset + 8])[0]
              
            current_offset += size
        
        elif operand.qualifier == OperandQualifier.REG:
          # Register index
          if current_offset < len(current_section.data):
            operand.value = current_section.data[current_offset]
            current_offset += 1
            
        elif operand.qualifier == OperandQualifier.MEM:
          # Memory address (32-bit)
          if current_offset + 4 <= len(current_section.data):
            operand.value = struct.unpack('<I', current_section.data[current_offset:current_offset + 4])[0]
            current_offset += 4
            
        elif operand.qualifier == OperandQualifier.LBL or operand.qualifier == OperandQualifier.REL:
          # Label or relative offset (32-bit)
          if current_offset + 4 <= len(current_section.data):
            operand.value = struct.unpack('<i', current_section.data[current_offset:current_offset + 4])[0]
            current_offset += 4
            
        instruction.operands.append(operand)
      
      # Update instruction pointer for next instruction
      self.ip += (current_offset - section_offset)
      self.current_instruction_index += 1
      
      return instruction
    except Exception as e:
      print(f"Error decoding instruction at {hex(self.ip)}: {e}")
      self.ip += 1  # Skip this byte and try again
      return None
    
  def _execute(self, instruction: Instruction) -> None:
    """Execute the given instruction."""
    if instruction.opcode == Opcode.NOP:
      # No operation
      pass
      
    elif instruction.opcode == Opcode.HLT:
      # Halt execution
      self.halted = True
      
    elif instruction.opcode == Opcode.MOV or instruction.opcode == Opcode.MOVI:
      # Move data between registers or load immediate
      if len(instruction.operands) >= 2:
        dest = instruction.operands[0]
        src = instruction.operands[1]
        
        if dest.qualifier == OperandQualifier.REG and src.qualifier == OperandQualifier.REG:
          # Register to register
          self.registers[dest.value] = self.registers[src.value]
          
        elif dest.qualifier == OperandQualifier.REG and src.qualifier == OperandQualifier.IMM:
          # Immediate to register
          self.registers[dest.value] = src.value
          
        elif dest.qualifier == OperandQualifier.REG and src.qualifier == OperandQualifier.MEM:
          # Memory to register
          addr = src.value
          if 0 <= addr < len(self.memory):
            size = dest.type_width // 8 if dest.type_width else 8
            self.registers[dest.value] = int.from_bytes(self.memory[addr:addr + size], byteorder='little')
            
        elif dest.qualifier == OperandQualifier.MEM and src.qualifier == OperandQualifier.REG:
          # Register to memory
          addr = dest.value
          if 0 <= addr < len(self.memory) - 8:
            size = src.type_width // 8 if src.type_width else 8
            value_bytes = self.registers[src.value].to_bytes(size, byteorder='little')
            self.memory[addr:addr + size] = value_bytes
      
    elif instruction.opcode == Opcode.ADD:
      # Addition
      if len(instruction.operands) >= 3:
        dest = instruction.operands[0]
        src1 = instruction.operands[1]
        src2 = instruction.operands[2]
        
        if dest.qualifier == OperandQualifier.REG and src1.qualifier == OperandQualifier.REG and src2.qualifier == OperandQualifier.REG:
          # Register + Register -> Register
          result = self.registers[src1.value] + self.registers[src2.value]
          self.registers[dest.value] = result
          
          # Set flags
          self._update_flags(result)
          
        elif dest.qualifier == OperandQualifier.REG and src1.qualifier == OperandQualifier.REG and src2.qualifier == OperandQualifier.IMM:
          # Register + Immediate -> Register
          result = self.registers[src1.value] + src2.value
          self.registers[dest.value] = result
          
          # Set flags
          self._update_flags(result)
    
    elif instruction.opcode == Opcode.SUB:
      # Subtraction
      if len(instruction.operands) >= 3:
        dest = instruction.operands[0]
        src1 = instruction.operands[1]
        src2 = instruction.operands[2]
        
        if dest.qualifier == OperandQualifier.REG and src1.qualifier == OperandQualifier.REG and src2.qualifier == OperandQualifier.REG:
          # Register - Register -> Register
          result = self.registers[src1.value] - self.registers[src2.value]
          self.registers[dest.value] = result
          
          # Set flags
          self._update_flags(result)
          
        elif dest.qualifier == OperandQualifier.REG and src1.qualifier == OperandQualifier.REG and src2.qualifier == OperandQualifier.IMM:
          # Register - Immediate -> Register
          result = self.registers[src1.value] - src2.value
          self.registers[dest.value] = result
          
          # Set flags
          self._update_flags(result)
    
    elif instruction.opcode == Opcode.MUL:
      # Multiplication
      if len(instruction.operands) >= 3:
        dest = instruction.operands[0]
        src1 = instruction.operands[1]
        src2 = instruction.operands[2]
        
        if dest.qualifier == OperandQualifier.REG and src1.qualifier == OperandQualifier.REG and src2.qualifier == OperandQualifier.REG:
          # Register * Register -> Register
          result = self.registers[src1.value] * self.registers[src2.value]
          self.registers[dest.value] = result
          
          # Set flags
          self._update_flags(result)
          
        elif dest.qualifier == OperandQualifier.REG and src1.qualifier == OperandQualifier.REG and src2.qualifier == OperandQualifier.IMM:
          # Register * Immediate -> Register
          result = self.registers[src1.value] * src2.value
          self.registers[dest.value] = result
          
          # Set flags
          self._update_flags(result)
    
    elif instruction.opcode == Opcode.DIV:
      # Division
      if len(instruction.operands) >= 3:
        dest = instruction.operands[0]
        src1 = instruction.operands[1]
        src2 = instruction.operands[2]
        
        if dest.qualifier == OperandQualifier.REG and src1.qualifier == OperandQualifier.REG and src2.qualifier == OperandQualifier.REG:
          # Register / Register -> Register
          divisor = self.registers[src2.value]
          if divisor == 0:
            raise ZeroDivisionError("Division by zero")
            
          result = self.registers[src1.value] // divisor
          self.registers[dest.value] = result
          
          # Set flags
          self._update_flags(result)
          
        elif dest.qualifier == OperandQualifier.REG and src1.qualifier == OperandQualifier.REG and src2.qualifier == OperandQualifier.IMM:
          # Register / Immediate -> Register
          if src2.value == 0:
            raise ZeroDivisionError("Division by zero")
            
          result = self.registers[src1.value] // src2.value
          self.registers[dest.value] = result
          
          # Set flags
          self._update_flags(result)
    
    elif instruction.opcode == Opcode.CMP:
      # Compare
      if len(instruction.operands) >= 2:
        src1 = instruction.operands[0]
        src2 = instruction.operands[1]
        
        if src1.qualifier == OperandQualifier.REG and src2.qualifier == OperandQualifier.REG:
          # Compare Register with Register
          result = self.registers[src1.value] - self.registers[src2.value]
          
          # Set flags based on comparison
          self._update_flags(result)
          
        elif src1.qualifier == OperandQualifier.REG and src2.qualifier == OperandQualifier.IMM:
          # Compare Register with Immediate
          result = self.registers[src1.value] - src2.value
          
          # Set flags based on comparison
          self._update_flags(result)
    
    elif instruction.opcode == Opcode.BR:
      # Unconditional branch
      if len(instruction.operands) >= 1:
        target = instruction.operands[0]
        
        if target.qualifier == OperandQualifier.LBL:
          # Absolute address
          self.ip = target.value
          
        elif target.qualifier == OperandQualifier.REL:
          # Relative address
          self.ip += target.value
    
    elif instruction.opcode == Opcode.BRC:
      # Conditional branch
      if len(instruction.operands) >= 1:
        target = instruction.operands[0]
        condition = BranchCondition(instruction.qualifier)
        
        # Check if condition is met
        should_branch = self._check_condition(condition)
        
        if should_branch:
          if target.qualifier == OperandQualifier.LBL:
            # Absolute address
            self.ip = target.value
            
          elif target.qualifier == OperandQualifier.REL:
            # Relative address
            self.ip += target.value
    
    elif instruction.opcode == Opcode.CALL:
      # Call subroutine
      if len(instruction.operands) >= 1:
        target = instruction.operands[0]
        
        # Push return address onto stack
        return_addr = self.ip
        self._push_value(return_addr, 8)
        
        # Jump to target
        if target.qualifier == OperandQualifier.LBL:
          # Absolute address
          self.ip = target.value
          
        elif target.qualifier == OperandQualifier.REL:
          # Relative address
          self.ip += target.value
    
    elif instruction.opcode == Opcode.RET:
      # Return from subroutine
      
      # Pop return address from stack
      return_addr = self._pop_value(8)
      
      # Jump to return address
      self.ip = return_addr
    
    elif instruction.opcode == Opcode.PUSH:
      # Push onto stack
      if len(instruction.operands) >= 1:
        src = instruction.operands[0]
        
        if src.qualifier == OperandQualifier.REG:
          # Push register value
          size = src.type_width // 8 if src.type_width else 8
          self._push_value(self.registers[src.value], size)
          
        elif src.qualifier == OperandQualifier.IMM:
          # Push immediate value
          size = src.type_width // 8 if src.type_width else 4
          self._push_value(src.value, size)
    
    elif instruction.opcode == Opcode.POP:
      # Pop from stack
      if len(instruction.operands) >= 1:
        dest = instruction.operands[0]
        
        if dest.qualifier == OperandQualifier.REG:
          # Pop to register
          size = dest.type_width // 8 if dest.type_width else 8
          value = self._pop_value(size)
          self.registers[dest.value] = value
    
    elif instruction.opcode == Opcode.INC:
      # Increment
      if len(instruction.operands) >= 1:
        dest = instruction.operands[0]
        
        if dest.qualifier == OperandQualifier.REG:
          # Increment register
          self.registers[dest.value] += 1
          
          # Set flags
          self._update_flags(self.registers[dest.value])
    
    elif instruction.opcode == Opcode.DEC:
      # Decrement
      if len(instruction.operands) >= 1:
        dest = instruction.operands[0]
        
        if dest.qualifier == OperandQualifier.REG:
          # Decrement register
          self.registers[dest.value] -= 1
          
          # Set flags
          self._update_flags(self.registers[dest.value])
    
    elif instruction.opcode == Opcode.SYSC:
      # System call
      if len(instruction.operands) >= 1:
        syscall_num = instruction.operands[0]
        
        if syscall_num.qualifier == OperandQualifier.IMM:
          # Handle system call
          self._handle_syscall(syscall_num.value)
          
        elif syscall_num.qualifier == OperandQualifier.REG:
          # Handle system call with number in register
          self._handle_syscall(self.registers[syscall_num.value])
    
    elif instruction.opcode == Opcode.AND:
      # Bitwise AND
      if len(instruction.operands) >= 3:
        dest = instruction.operands[0]
        src1 = instruction.operands[1]
        src2 = instruction.operands[2]
        
        if dest.qualifier == OperandQualifier.REG and src1.qualifier == OperandQualifier.REG and src2.qualifier == OperandQualifier.REG:
          # Register & Register -> Register
          result = self.registers[src1.value] & self.registers[src2.value]
          self.registers[dest.value] = result
          
          # Set flags
          self._update_flags(result)
          
        elif dest.qualifier == OperandQualifier.REG and src1.qualifier == OperandQualifier.REG and src2.qualifier == OperandQualifier.IMM:
          # Register & Immediate -> Register
          result = self.registers[src1.value] & src2.value
          self.registers[dest.value] = result
          
          # Set flags
          self._update_flags(result)
    
    elif instruction.opcode == Opcode.OR:
      # Bitwise OR
      if len(instruction.operands) >= 3:
        dest = instruction.operands[0]
        src1 = instruction.operands[1]
        src2 = instruction.operands[2]
        
        if dest.qualifier == OperandQualifier.REG and src1.qualifier == OperandQualifier.REG and src2.qualifier == OperandQualifier.REG:
          # Register | Register -> Register
          result = self.registers[src1.value] | self.registers[src2.value]
          self.registers[dest.value] = result
          
          # Set flags
          self._update_flags(result)
          
        elif dest.qualifier == OperandQualifier.REG and src1.qualifier == OperandQualifier.REG and src2.qualifier == OperandQualifier.IMM:
          # Register | Immediate -> Register
          result = self.registers[src1.value] | src2.value
          self.registers[dest.value] = result
          
          # Set flags
          self._update_flags(result)
    
    elif instruction.opcode == Opcode.XOR:
      # Bitwise XOR
      if len(instruction.operands) >= 3:
        dest = instruction.operands[0]
        src1 = instruction.operands[1]
        src2 = instruction.operands[2]
        
        if dest.qualifier == OperandQualifier.REG and src1.qualifier == OperandQualifier.REG and src2.qualifier == OperandQualifier.REG:
          # Register ^ Register -> Register
          result = self.registers[src1.value] ^ self.registers[src2.value]
          self.registers[dest.value] = result
          
          # Set flags
          self._update_flags(result)
          
        elif dest.qualifier == OperandQualifier.REG and src1.qualifier == OperandQualifier.REG and src2.qualifier == OperandQualifier.IMM:
          # Register ^ Immediate -> Register
          result = self.registers[src1.value] ^ src2.value
          self.registers[dest.value] = result
          
          # Set flags
          self._update_flags(result)
    
    # Handle function entry/exit
    elif instruction.opcode == Opcode.ENTER:
      # Function prologue
      # Push base pointer
      self._push_value(self.bp, 8)
      # Set base pointer to current stack pointer
      self.bp = self.sp
      # Allocate stack space for local variables if needed
      if len(instruction.operands) >= 1 and instruction.operands[0].qualifier == OperandQualifier.IMM:
        # Allocate stack space
        self.sp -= instruction.operands[0].value
    
    elif instruction.opcode == Opcode.LEAVE:
      # Function epilogue
      # Restore stack pointer from base pointer
      self.sp = self.bp
      # Restore base pointer
      self.bp = self._pop_value(8)
    
    # Add more instruction implementations as needed
    
  def _update_flags(self, result: int) -> None:
    """Update flags register based on result."""
    # Truncate to 64 bits
    result = result & 0xFFFFFFFFFFFFFFFF
    
    # Zero flag
    self.flags['zero'] = (result == 0)
    
    # Sign flag (bit 63 for 64-bit value)
    self.flags['sign'] = ((result & 0x8000000000000000) != 0)
    
    # Parity flag (1 if even number of 1 bits)
    parity = bin(result).count('1') % 2 == 0
    self.flags['parity'] = parity
    
    # Note: carry and overflow flags are set in the actual operations
  
  def _check_condition(self, condition: BranchCondition) -> bool:
    """Check if a branch condition is met."""
    if condition == BranchCondition.ALWAYS:
      return True
    elif condition == BranchCondition.EQ:
      return self.flags['zero']
    elif condition == BranchCondition.NE:
      return not self.flags['zero']
    elif condition == BranchCondition.LT:
      return self.flags['sign'] and not self.flags['zero']
    elif condition == BranchCondition.LE:
      return self.flags['sign'] or self.flags['zero']
    elif condition == BranchCondition.GT:
      return not self.flags['sign'] and not self.flags['zero']
    elif condition == BranchCondition.GE:
      return not self.flags['sign'] or self.flags['zero']
    elif condition == BranchCondition.CARRY:
      return self.flags['carry']
    elif condition == BranchCondition.OFLOW:
      return self.flags['overflow']
    elif condition == BranchCondition.SIGN:
      return self.flags['sign']
    elif condition == BranchCondition.PARITY:
      return self.flags['parity']
    else:
      return False
  
  def _push_value(self, value: int, size: int) -> None:
    """Push a value onto the stack."""
    # Adjust stack pointer
    self.sp -= size
    
    # Check stack bounds
    if self.sp < 0:
      raise OverflowError("Stack overflow")
    
    # Write value to stack
    value_bytes = value.to_bytes(size, byteorder='little')
    self.memory[self.sp:self.sp + size] = value_bytes
  
  def _pop_value(self, size: int) -> int:
    """Pop a value from the stack."""
    # Check stack bounds
    if self.sp + size > len(self.memory):
      raise IndexError("Stack underflow")
    
    # Read value from stack
    value_bytes = self.memory[self.sp:self.sp + size]
    value = int.from_bytes(value_bytes, byteorder='little')
    
    # Adjust stack pointer
    self.sp += size
    
    return value
  
  def _handle_syscall(self, syscall_num: int) -> None:
    """Handle system calls based on syscall number."""
    # For demonstration, implement a minimal subset of Linux-like syscalls
    
    if syscall_num == 1:  # write
      # args: fd, buffer, count
      fd = self.registers[0]
      buf_addr = self.registers[1]
      count = self.registers[2]
      
      # Only handle stdout (fd 1) for now
      if fd == 1:
        data = bytes(self.memory[buf_addr:buf_addr + count])
        sys.stdout.write(data.decode('utf-8', errors='ignore'))
        sys.stdout.flush()
        
        # Return number of bytes written
        self.registers[0] = count
      else:
        # Unsupported file descriptor
        self.registers[0] = -1
        
    elif syscall_num == 60:  # exit
      # args: exit_code
      exit_code = self.registers[0]
      print(f"Program exited with code {exit_code}")
      self.halted = True
      
    else:
      # Unsupported syscall
      print(f"Unsupported syscall: {syscall_num}")
      self.registers[0] = -1

# =============================================================================
# Main program
# =============================================================================

def main():
  parser = argparse.ArgumentParser(description='COIL Virtual Machine')
  parser.add_argument('file', help='COIL object file to execute')
  parser.add_argument('-d', '--debug', action='store_true', help='Enable debug output')
  
  args = parser.parse_args()
  
  # Create and initialize VM
  vm = CoilVM()
  vm.debug = args.debug
  
  # Load COIL object file
  if vm.load_file(args.file):
    # Run the program
    vm.run()
  else:
    sys.exit(1)

if __name__ == '__main__':
  main()