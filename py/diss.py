#!/usr/bin/env python3
"""
COIL Disassembler - A tool to convert COIL binary format to CEL-like representation
Version 0.2.0

This disassembler helps diagnose issues with COIL binary files by showing their 
structure and content in a human-readable format similar to CEL syntax.
"""

import struct
import sys
import os
import argparse
from enum import IntEnum
from typing import Dict, List, Tuple, Optional, BinaryIO

# =============================================================================
# Constants and Enumerations
# =============================================================================

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
  
  # Function entry/exit operations (0xC0 - 0xCF)
  ENTER = 0xC0  # Function prologue
  LEAVE = 0xC1  # Function epilogue
  PARAM = 0xC2  # Function parameter
  RESULT = 0xC3  # Function result
  ALLOCA = 0xC4  # Allocate stack memory
  
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
# Disassembler Class
# =============================================================================

class CoilDisassembler:
  def __init__(self, filename, verbose=False, cel_syntax=True):
    self.filename = filename
    self.verbose = verbose
    self.cel_syntax = cel_syntax
    self.file_data = bytearray()
    self.position = 0
    self.header = {}
    self.sections = []
    self.string_table = bytearray()
    self.symbol_table = []
    self.disassembly = []
    self.big_endian_magic = False
    
  def disassemble(self):
    self._read_file()
    self._parse_header()
    
    # Check if we have a valid COIL file (using either endianness)
    valid_coil = self.header.get('magic') == 0x434F494C  # 'COIL'
    
    if valid_coil:
      self._parse_sections()
      
      # If there's an entrypoint, try to disassemble that area
      entrypoint = self.header.get('entrypoint', 0)
      if entrypoint > 0 and entrypoint < len(self.file_data):
        print("\n=== ENTRYPOINT CODE ===")
        print(f"Disassembling from entrypoint at 0x{entrypoint:08X}:")
        
        # Determine a reasonable size for disassembly (up to end of file or max 256 bytes)
        size = min(256, len(self.file_data) - entrypoint)
        self._disassemble_code_section(entrypoint, size, is_entrypoint=True)
      
      self._display_disassembly()
    else:
      print(f"Warning: Not a valid COIL file. Magic number: {hex(self.header.get('magic', 0))}")
      self._raw_dump()
  
  def _read_file(self):
    """Read the entire file into memory."""
    try:
      with open(self.filename, 'rb') as f:
        self.file_data = bytearray(f.read())
      print(f"File size: {len(self.file_data)} bytes")
    except Exception as e:
      print(f"Error reading file: {e}")
      sys.exit(1)
  
  def _parse_header(self):
    """Attempt to parse the COF header, with fallbacks for incomplete headers."""
    print("\n=== COIL HEADER ===")
    
    # Check if we have enough bytes for a header
    if len(self.file_data) < 32:
      print(f"Warning: File too small for complete header. Only {len(self.file_data)} bytes available.")
      
      # Try to parse what we can
      if len(self.file_data) >= 4:
        magic_le = struct.unpack('<I', self.file_data[0:4])[0]
        magic_be = struct.unpack('>I', self.file_data[0:4])[0]
        
        # Check for magic number in either endianness
        if magic_be == 0x434F494C:  # 'COIL'
          print(f"Magic number (big-endian): {hex(magic_be)} ('COIL')")
          self.header['magic'] = magic_be
          self.big_endian_magic = True
        else:
          print(f"Magic number: {hex(magic_le)} (Invalid - expected 'COIL')")
          self.header['magic'] = magic_le
      
      if len(self.file_data) >= 7:
        version = struct.unpack('<BBB', self.file_data[4:7])
        print(f"Version: {version[0]}.{version[1]}.{version[2]}")
        self.header['version_major'] = version[0]
        self.header['version_minor'] = version[1]
        self.header['version_patch'] = version[2]
      
      # Dump remaining header bytes as raw data
      remaining = min(32, len(self.file_data))
      print("\nRaw header bytes:")
      for i in range(0, remaining, 4):
        chunk = self.file_data[i:min(i+4, remaining)]
        hex_str = ' '.join(f'{b:02X}' for b in chunk)
        print(f"  {i:04X}: {hex_str}")
      
      return
    
    # Parse the complete header
    try:
      header_data = self.file_data[0:32]
      
      # Check for both little-endian and big-endian magic numbers
      magic_le = struct.unpack('<I', header_data[0:4])[0]
      magic_be = struct.unpack('>I', header_data[0:4])[0]
      
      if magic_be == 0x434F494C:  # 'COIL'
        print("Detected big-endian magic number ('COIL' as 'LIOC'), adjusting interpretation.")
        magic = magic_be
        self.big_endian_magic = True
      else:
        magic = magic_le
      
      # Parse the rest of the header (using little-endian)
      version_major, version_minor, version_patch, flags, target, section_count = struct.unpack('<BBBBHH', header_data[4:12])
      entrypoint, str_tab_off, str_tab_size, sym_tab_off, sym_tab_size = struct.unpack('<IIIII', header_data[12:32])
      
      self.header = {
        'magic': magic,
        'version_major': version_major,
        'version_minor': version_minor,
        'version_patch': version_patch,
        'flags': flags,
        'target': target,
        'section_count': section_count,
        'entrypoint': entrypoint,
        'str_tab_off': str_tab_off,
        'str_tab_size': str_tab_size,
        'sym_tab_off': sym_tab_off,
        'sym_tab_size': sym_tab_size
      }
      
      # Display header information
      print(f"Magic number: 0x{magic:08X} ({'COIL' if magic == 0x434F494C else 'INVALID'})")
      print(f"Version: {version_major}.{version_minor}.{version_patch}")
      print(f"Flags: 0x{flags:02X}")
      print(f"Target architecture: 0x{target:04X}")
      print(f"Section count: {section_count}")
      print(f"Entrypoint offset: 0x{entrypoint:08X}")
      print(f"String table offset: 0x{str_tab_off:08X}, size: {str_tab_size} bytes")
      print(f"Symbol table offset: 0x{sym_tab_off:08X}, size: {sym_tab_size} bytes")
      
      self.position = 32  # Move past header
      
    except Exception as e:
      print(f"Error parsing header: {e}")
      print("\nRaw header bytes:")
      for i in range(0, min(32, len(self.file_data)), 4):
        chunk = self.file_data[i:min(i+4, len(self.file_data))]
        hex_str = ' '.join(f'{b:02X}' for b in chunk)
        print(f"  {i:04X}: {hex_str}")
  
  def _parse_sections(self):
    """Parse section headers and data."""
    print("\n=== SECTIONS ===")
    
    section_count = self.header.get('section_count', 0)
    if section_count == 0:
      print("No sections found.")
      return
    
    # Load string table first if available
    str_tab_off = self.header.get('str_tab_off', 0)
    str_tab_size = self.header.get('str_tab_size', 0)
    
    if str_tab_off > 0 and str_tab_size > 0 and str_tab_off + str_tab_size <= len(self.file_data):
      self.string_table = self.file_data[str_tab_off:str_tab_off + str_tab_size]
      print(f"String table loaded: {str_tab_size} bytes at offset 0x{str_tab_off:08X}")
    
    # Try to parse section headers
    for i in range(section_count):
      if self.position + 36 > len(self.file_data):
        print(f"Warning: File truncated. Can't read section header {i+1}.")
        break
      
      section_header_data = self.file_data[self.position:self.position + 36]
      
      try:
        name_offset, type_val, flags, offset, size, link, info, alignment, entsize = \
          struct.unpack('<IIIIIIIII', section_header_data)
        
        section_name = self._get_string(name_offset)
        section_type = self._get_section_type_name(type_val)
        
        print(f"\nSection {i+1}: {section_name}")
        print(f"  Type: {section_type} ({type_val})")
        print(f"  Flags: 0x{flags:08X}")
        print(f"  Offset: 0x{offset:08X}, Size: {size} bytes")
        print(f"  Link: {link}, Info: {info}")
        print(f"  Alignment: {alignment}, Entry size: {entsize}")
        
        # Store section information
        self.sections.append({
          'name': section_name,
          'type': type_val,
          'flags': flags,
          'offset': offset,
          'size': size,
          'link': link,
          'info': info,
          'alignment': alignment,
          'entsize': entsize
        })
        
        # Disassemble code sections, or any section with an offset > 0 and size > 0
        if type_val == SectionType.CODE:
          if offset > 0 and size > 0:
            self._disassemble_code_section(offset, size)
        
        # Try to disassemble any non-NULL section that has data
        elif type_val != SectionType.NULL and offset > 0 and size > 0:
          # Check if first byte could be an opcode
          if offset < len(self.file_data) and self.file_data[offset] < 0xE0:
            print("\n  Section might contain code, attempting disassembly:")
            self._disassemble_code_section(offset, size)
            
        # Always dump section data if non-empty
        if offset > 0 and size > 0:
          self._dump_data_section(offset, size)
        
        # Parse symbol table
        if type_val == SectionType.SYMTAB and offset > 0 and size > 0:
          self._parse_symbol_table(offset, size, entsize)
        
        self.position += 36  # Move to next section header
        
      except Exception as e:
        print(f"Error parsing section header: {e}")
        self.position += 36  # Try to recover by skipping this header
    
  def _get_string(self, offset):
    """Get a null-terminated string from the string table."""
    if not self.string_table or offset >= len(self.string_table):
      return f"<string at offset {offset}>"
    
    end = self.string_table.find(0, offset)
    if end == -1:
      return self.string_table[offset:].decode('utf-8', errors='replace')
    else:
      return self.string_table[offset:end].decode('utf-8', errors='replace')
  
  def _get_section_type_name(self, type_val):
    """Get the name of a section type."""
    try:
      return SectionType(type_val).name
    except:
      return f"UNKNOWN({type_val})"
  
  def _disassemble_code_section(self, offset, size, is_entrypoint=False):
    """Disassemble a code section into CEL-like instructions."""
    if not is_entrypoint:
      print("\n  Disassembly:")
    
    if offset + size > len(self.file_data):
      print("  Warning: Section extends beyond file size. Truncating.")
      size = len(self.file_data) - offset
    
    section_data = self.file_data[offset:offset + size]
    pos = 0
    
    while pos < size:
      # Save the original position for address calculation
      original_pos = pos
      
      # Make sure we have at least 3 bytes for an instruction (opcode, qualifier, operand count)
      if pos + 3 > size:
        print(f"  {offset + pos:08X}: Incomplete instruction")
        self._dump_bytes(section_data[pos:])
        break
      
      # Read basic instruction components
      opcode_val = section_data[pos]
      qualifier_val = section_data[pos + 1]
      operand_count = section_data[pos + 2]
      pos += 3
      
      # Try to get opcode name
      try:
        if 0xD0 <= opcode_val <= 0xDF:
          opcode_name = f"DIR_{Opcode(opcode_val).name}" if hasattr(Opcode, str(opcode_val)) else f"DIR_0x{opcode_val:02X}"
        else:
          opcode_name = Opcode(opcode_val).name if hasattr(Opcode, str(opcode_val)) else f"0x{opcode_val:02X}"
      except:
        opcode_name = f"0x{opcode_val:02X}"
      
      # Special handling for branch conditions
      if opcode_val == Opcode.BRC:
        try:
          qualifier_name = BranchCondition(qualifier_val).name
        except:
          qualifier_name = f"0x{qualifier_val:02X}"
      else:
        qualifier_name = f"0x{qualifier_val:02X}" if qualifier_val != 0 else ""
      
      # Start building the instruction string
      if self.cel_syntax:
        instr_str = f"{opcode_name}"
        if qualifier_val != 0:
          instr_str += f" {qualifier_name}"
      else:
        instr_str = f"{opcode_name}"
        if qualifier_val != 0:
          instr_str += f" {qualifier_name}"
      
      # Parse operands
      operands = []
      cel_operands = []
      for i in range(operand_count):
        if pos + 2 > size:
          operands.append("<incomplete>")
          cel_operands.append("<incomplete>")
          break
        
        operand_qual = section_data[pos]
        operand_type = section_data[pos + 1]
        pos += 2
        
        # Get qualifier and type names
        try:
          qual_name = OperandQualifier(operand_qual).name
        except:
          qual_name = f"Q{operand_qual:02X}"
        
        try:
          type_name = CoilType(operand_type).name
        except:
          type_name = f"T{operand_type:02X}"
        
        # Handle type width for certain types
        type_width = None
        if operand_type in [CoilType.INT, CoilType.UINT, CoilType.FLOAT] and pos < size:
          type_width = section_data[pos]
          pos += 1
        
        # Process operand value based on qualifier
        value_str = ""
        cel_value_str = ""
        
        if operand_qual == OperandQualifier.REG:
          # Register
          if pos < size:
            reg_num = section_data[pos]
            pos += 1
            
            # Get register name based on width
            prefix = "R"
            if type_width is not None:
              if type_width == 8:
                prefix += "B"
              elif type_width == 16:
                prefix += "W"
              elif type_width == 32:
                prefix += "L"
              elif type_width == 64:
                prefix += "Q"
              else:
                prefix += f"_{type_width}"
            else:
              prefix += "Q"  # Default to 64-bit
            
            value_str = f"{prefix}{reg_num}"
            cel_value_str = value_str
          else:
            value_str = "<incomplete>"
            cel_value_str = value_str
        
        elif operand_qual == OperandQualifier.IMM:
          # Immediate value
          width = type_width if type_width is not None else 32
          size_bytes = width // 8
          
          if pos + size_bytes <= size:
            if width == 8:
              value = section_data[pos]
              value_str = f"0x{value:02X}"
              cel_value_str = str(value)
            elif width == 16:
              value = struct.unpack('<H', section_data[pos:pos + 2])[0]
              value_str = f"0x{value:04X}"
              cel_value_str = str(value)
            elif width == 32:
              value = struct.unpack('<I', section_data[pos:pos + 4])[0]
              value_str = f"0x{value:08X}"
              cel_value_str = str(value)
            elif width == 64:
              value = struct.unpack('<Q', section_data[pos:pos + 8])[0]
              value_str = f"0x{value:016X}"
              cel_value_str = str(value)
            
            pos += size_bytes
          else:
            value_str = "<incomplete>"
            cel_value_str = value_str
        
        elif operand_qual == OperandQualifier.MEM:
          # Memory address
          if pos + 4 <= size:
            addr = struct.unpack('<I', section_data[pos:pos + 4])[0]
            value_str = f"[0x{addr:08X}]"
            cel_value_str = f"[{addr}]"
            pos += 4
          else:
            value_str = "<incomplete>"
            cel_value_str = value_str
        
        elif operand_qual == OperandQualifier.LBL:
          # Label
          if pos + 4 <= size:
            addr = struct.unpack('<I', section_data[pos:pos + 4])[0]
            value_str = f"LBL_0x{addr:08X}"
            cel_value_str = f"label_{addr:X}"
            pos += 4
          else:
            value_str = "<incomplete>"
            cel_value_str = value_str
        
        elif operand_qual == OperandQualifier.REL:
          # Relative offset
          if pos + 4 <= size:
            offset_val = struct.unpack('<i', section_data[pos:pos + 4])[0]  # Signed
            value_str = f"REL{'+' if offset_val >= 0 else ''}{offset_val}"
            cel_value_str = f"{offset_val:+}"
            pos += 4
          else:
            value_str = "<incomplete>"
            cel_value_str = value_str
        
        else:
          # Other qualifiers
          if self.cel_syntax:
            value_str = f"{qual_name}:{type_name}"
            cel_value_str = value_str
          else:
            value_str = f"{qual_name}:{type_name}"
            cel_value_str = value_str
          
          # Skip any remaining operand data
          pos += 4  # Assume a reasonable default size
        
        operands.append(value_str)
        cel_operands.append(cel_value_str)
      
      # Complete the instruction string
      if operands:
        if self.cel_syntax:
          instr_str += " " + ", ".join(cel_operands)
        else:
          instr_str += " " + ", ".join(operands)
      
      # Calculate instruction size
      instr_size = pos - original_pos
      
      # Get bytes for this instruction
      instr_bytes = section_data[original_pos:pos]
      bytes_str = ' '.join(f'{b:02X}' for b in instr_bytes)
      
      # Add to disassembly
      addr_str = f"{offset + original_pos:08X}"
      
      if self.cel_syntax:
        # Format as CEL-like syntax
        disasm_line = f"  {addr_str}: {bytes_str.ljust(24)} ; {instr_str}"
      else:
        # Format with raw hex values
        disasm_line = f"  {addr_str}: {bytes_str.ljust(24)} {instr_str}"
      
      print(disasm_line)
      self.disassembly.append(disasm_line)
  
  def _dump_data_section(self, offset, size):
    """Dump the contents of a data section."""
    print("\n  Data section contents:")
    
    if offset + size > len(self.file_data):
      print("  Warning: Section extends beyond file size. Truncating.")
      size = len(self.file_data) - offset
    
    section_data = self.file_data[offset:offset + size]
    
    for i in range(0, size, 16):
      chunk = section_data[i:min(i+16, size)]
      hex_str = ' '.join(f'{b:02X}' for b in chunk)
      
      # Try to interpret as ASCII
      ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
      
      print(f"  {offset + i:08X}: {hex_str.ljust(48)} |{ascii_str}|")
  
  def _parse_symbol_table(self, offset, size, entsize):
    """Parse the symbol table."""
    print("\n  Symbol Table:")
    
    if offset + size > len(self.file_data):
      print("  Warning: Symbol table extends beyond file size. Truncating.")
      size = len(self.file_data) - offset
    
    # Use the entry size from the section header, or default to 16 bytes
    entry_size = entsize if entsize > 0 else 16
    
    for i in range(0, size, entry_size):
      if offset + i + 16 > len(self.file_data):
        print("  Warning: Incomplete symbol table entry.")
        break
      
      try:
        name_offset, value, symbol_size, type_val, binding, visibility, section_index = \
          struct.unpack('<IIIBBBB', self.file_data[offset + i:offset + i + 16])
        
        sym_name = self._get_string(name_offset)
        
        # Get type and binding names
        type_name = f"TYPE_{type_val}"
        binding_name = f"BIND_{binding}"
        
        print(f"  Symbol: {sym_name}")
        print(f"    Value: 0x{value:08X}, Size: {symbol_size}")
        print(f"    Type: {type_name}, Binding: {binding_name}")
        print(f"    Visibility: {visibility}, Section: {section_index}")
        
        # Store symbol
        self.symbol_table.append({
          'name': sym_name,
          'value': value,
          'size': symbol_size,
          'type': type_val,
          'binding': binding,
          'visibility': visibility,
          'section_index': section_index
        })
        
      except Exception as e:
        print(f"  Error parsing symbol: {e}")
  
  def _dump_bytes(self, data):
    """Dump a byte sequence in hex."""
    hex_str = ' '.join(f'{b:02X}' for b in data)
    print(f"  Bytes: {hex_str}")
  
  def _raw_dump(self):
    """Perform a raw hex dump of the file."""
    print("\n=== RAW HEX DUMP ===")
    
    for i in range(0, len(self.file_data), 16):
      chunk = self.file_data[i:min(i+16, len(self.file_data))]
      hex_str = ' '.join(f'{b:02X}' for b in chunk)
      
      # Try to interpret as ASCII
      ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
      
      print(f"{i:08X}: {hex_str.ljust(48)} |{ascii_str}|")
  
  def _display_disassembly(self):
    """Display summary of the disassembly."""
    if self.header.get('magic') != 0x434F494C:
      print("\n=== WARNING: Not a valid COIL file ===")
    
    print(f"\n=== SUMMARY ===")
    print(f"File: {self.filename}")
    print(f"COIL Version: {self.header.get('version_major', '?')}.{self.header.get('version_minor', '?')}.{self.header.get('version_patch', '?')}")
    if self.big_endian_magic:
      print("Note: Using big-endian magic number ('COIL' as 'LIOC')")
    print(f"Sections: {len(self.sections)}")
    print(f"Symbols: {len(self.symbol_table)}")
    
    # List section names
    if self.sections:
      print("\nSections:")
      for i, section in enumerate(self.sections):
        print(f"  {i+1}: {section['name']} ({self._get_section_type_name(section['type'])})")

    # Print CEL header if using CEL syntax
    if self.cel_syntax and self.disassembly:
      entrypoint = self.header.get('entrypoint', 0)
      if entrypoint > 0:
        print("\n=== CEL REPRESENTATION ===")
        print(".version 1.0.0")
        print(f".target {'x86-64' if self.header.get('target') == 2 else 'unknown'}")
        print(".section .text, \"x\"")
        print("")
        print("main:")
        # No need to repeat the disassembly here
        print("  ; See disassembly above for contents")

# =============================================================================
# Main program
# =============================================================================

def main():
  parser = argparse.ArgumentParser(description='COIL Disassembler')
  parser.add_argument('file', help='COIL object file to disassemble')
  parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
  parser.add_argument('-c', '--cel', action='store_true', help='Format output in CEL-like syntax', default=True)
  
  args = parser.parse_args()
  
  disassembler = CoilDisassembler(args.file, args.verbose, args.cel)
  disassembler.disassemble()

if __name__ == '__main__':
  main()