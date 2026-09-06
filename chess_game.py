"""
Chess Master AI — Human vs AI Chess Game
Minimax with Alpha-Beta Pruning + Greedy Evaluation
"""

import pygame
import sys
import copy
import time
import random
from typing import List, Tuple, Optional, Dict

# ─── Constants ────────────────────────────────────────────────────────────────

BOARD_SIZE = 8
SQUARE_SIZE = 90
BOARD_PX = BOARD_SIZE * SQUARE_SIZE
SIDEBAR_WIDTH = 220
WINDOW_WIDTH = BOARD_PX + SIDEBAR_WIDTH
WINDOW_HEIGHT = BOARD_PX + 80  # extra for top bar

FPS = 60

# Colors — warm, natural palette (not AI-typical)
LIGHT_SQ = (232, 210, 176)      # warm cream
DARK_SQ = (181, 136, 99)        # walnut brown
HIGHLIGHT_SRC = (247, 247, 105, 180)  # soft yellow overlay
HIGHLIGHT_DST = (105, 247, 152, 120)  # soft green overlay
HIGHLIGHT_LAST = (205, 210, 106, 100) # muted olive for last move
CHECK_COLOR = (235, 64, 52, 160)      # warm red for check

BG_COLOR = (44, 40, 37)          # dark charcoal
SIDEBAR_BG = (56, 52, 48)        # slightly lighter charcoal
TEXT_COLOR = (220, 215, 205)     # off-white
ACCENT_COLOR = (210, 170, 110)   # golden accent
BUTTON_COLOR = (80, 74, 68)
BUTTON_HOVER = (100, 92, 84)
BUTTON_TEXT = (230, 225, 215)

# Piece values for evaluation
PIECE_VALUES = {
    'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000,
    'p': 100, 'n': 320, 'b': 330, 'r': 500, 'q': 900, 'k': 20000,
}

# Positional bonus tables (from White's perspective, mirrored for Black)
PST_PAWN = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]

PST_KNIGHT = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]

PST_BISHOP = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]

PST_ROOK = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
]

PST_QUEEN = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
]

PST_KING_MID = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
]

PST = {
    'P': PST_PAWN, 'N': PST_KNIGHT, 'B': PST_BISHOP,
    'R': PST_ROOK, 'Q': PST_QUEEN, 'K': PST_KING_MID,
    'p': PST_PAWN, 'n': PST_KNIGHT, 'b': PST_BISHOP,
    'r': PST_ROOK, 'q': PST_QUEEN, 'k': PST_KING_MID,
}

# Unicode piece symbols
UNICODE_PIECES = {
    'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
    'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟',
}

# AI depth settings
AI_DEPTH = 3


# ─── Chess Engine ─────────────────────────────────────────────────────────────

class ChessBoard:
    def __init__(self):
        self.board = [
            ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'],
            ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
            ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R'],
        ]
        self.white_to_move = True
        self.castling_rights = {'K': True, 'Q': True, 'k': True, 'q': True}
        self.en_passant_sq: Optional[Tuple[int, int]] = None
        self.halfmove_clock = 0
        self.fullmove_number = 1
        self.move_log: List = []

    def copy(self):
        b = ChessBoard()
        b.board = [row[:] for row in self.board]
        b.white_to_move = self.white_to_move
        b.castling_rights = dict(self.castling_rights)
        b.en_passant_sq = self.en_passant_sq
        b.halfmove_clock = self.halfmove_clock
        b.fullmove_number = self.fullmove_number
        return b

    def piece_at(self, r, c):
        return self.board[r][c]

    def is_white(self, piece):
        return piece != '.' and piece.isupper()

    def is_black(self, piece):
        return piece != '.' and piece.islower()

    def is_enemy(self, piece, white_turn):
        if piece == '.':
            return False
        return self.is_black(piece) if white_turn else self.is_white(piece)

    def is_friendly(self, piece, white_turn):
        if piece == '.':
            return False
        return self.is_white(piece) if white_turn else self.is_black(piece)

    def in_bounds(self, r, c):
        return 0 <= r < 8 and 0 <= c < 8

    def find_king(self, white):
        k = 'K' if white else 'k'
        for r in range(8):
            for c in range(8):
                if self.board[r][c] == k:
                    return (r, c)
        return None

    def is_square_attacked(self, r, c, by_white):
        """Check if square (r,c) is attacked by the given side."""
        # Check from all opponent pieces
        for rr in range(8):
            for cc in range(8):
                p = self.board[rr][cc]
                if p == '.':
                    continue
                if by_white and not self.is_white(p):
                    continue
                if not by_white and not self.is_black(p):
                    continue

                # Check if this piece can attack (r,c)
                dr = r - rr
                dc = c - cc

                pt = p.upper()

                if pt == 'P':
                    direction = -1 if self.is_white(p) else 1
                    if dr == direction and abs(dc) == 1:
                        return True

                elif pt == 'N':
                    if (abs(dr), abs(dc)) in [(2, 1), (1, 2)]:
                        return True

                elif pt == 'K':
                    if abs(dr) <= 1 and abs(dc) <= 1:
                        return True

                elif pt == 'B':
                    if abs(dr) == abs(dc) and abs(dr) > 0:
                        step_r = dr // abs(dr)
                        step_c = dc // abs(dc)
                        blocked = False
                        for i in range(1, abs(dr)):
                            if self.board[rr + i * step_r][cc + i * step_c] != '.':
                                blocked = True
                                break
                        if not blocked:
                            return True

                elif pt == 'R':
                    if (dr == 0 or dc == 0) and (dr != 0 or dc != 0):
                        step_r = 0 if dr == 0 else (1 if dr > 0 else -1)
                        step_c = 0 if dc == 0 else (1 if dc > 0 else -1)
                        dist = max(abs(dr), abs(dc))
                        blocked = False
                        for i in range(1, dist):
                            if self.board[rr + i * step_r][cc + i * step_c] != '.':
                                blocked = True
                                break
                        if not blocked:
                            return True

                elif pt == 'Q':
                    if (abs(dr) == abs(dc) and abs(dr) > 0) or \
                       ((dr == 0 or dc == 0) and (dr != 0 or dc != 0)):
                        step_r = 0 if dr == 0 else (1 if dr > 0 else -1)
                        step_c = 0 if dc == 0 else (1 if dc > 0 else -1)
                        dist = max(abs(dr), abs(dc))
                        blocked = False
                        for i in range(1, dist):
                            if self.board[rr + i * step_r][cc + i * step_c] != '.':
                                blocked = True
                                break
                        if not blocked:
                            return True
        return False

    def in_check(self, white):
        king = self.find_king(white)
        if king is None:
            return True
        return self.is_square_attacked(king[0], king[1], not white)

    def generate_pseudo_legal_moves(self, white):
        """Generate all pseudo-legal moves (may leave king in check)."""
        moves = []
        for r in range(8):
            for c in range(8):
                p = self.board[r][c]
                if p == '.':
                    continue
                if white and not self.is_white(p):
                    continue
                if not white and not self.is_black(p):
                    continue

                pt = p.upper()

                if pt == 'P':
                    self._gen_pawn_moves(r, c, p, white, moves)
                elif pt == 'N':
                    self._gen_knight_moves(r, c, p, white, moves)
                elif pt == 'B':
                    self._gen_sliding_moves(r, c, p, white, moves,
                                            [(-1,-1),(-1,1),(1,-1),(1,1)])
                elif pt == 'R':
                    self._gen_sliding_moves(r, c, p, white, moves,
                                            [(-1,0),(1,0),(0,-1),(0,1)])
                elif pt == 'Q':
                    self._gen_sliding_moves(r, c, p, white, moves,
                                            [(-1,-1),(-1,1),(1,-1),(1,1),
                                             (-1,0),(1,0),(0,-1),(0,1)])
                elif pt == 'K':
                    self._gen_king_moves(r, c, p, white, moves)
        return moves

    def _gen_pawn_moves(self, r, c, p, white, moves):
        direction = -1 if white else 1
        start_row = 6 if white else 1
        promo_row = 0 if white else 7

        # Forward one
        nr = r + direction
        if self.in_bounds(nr, c) and self.board[nr][c] == '.':
            if nr == promo_row:
                for promo in ['Q', 'R', 'B', 'N']:
                    moves.append(('P', r, c, nr, c, 'promote', promo if white else promo.lower()))
            else:
                moves.append(('P', r, c, nr, c, 'normal'))

            # Forward two from start
            if r == start_row:
                nr2 = r + 2 * direction
                if self.board[nr2][c] == '.':
                    moves.append(('P', r, c, nr2, c, 'double_push'))

        # Captures
        for dc in [-1, 1]:
            nc = c + dc
            nr = r + direction
            if self.in_bounds(nr, nc):
                target = self.board[nr][nc]
                if self.is_enemy(target, white):
                    if nr == promo_row:
                        for promo in ['Q', 'R', 'B', 'N']:
                            moves.append(('P', r, c, nr, nc, 'capture_promote', promo if white else promo.lower()))
                    else:
                        moves.append(('P', r, c, nr, nc, 'capture'))

                # En passant
                if self.en_passant_sq == (nr, nc):
                    cap_r = r  # the captured pawn is on same row as capturing pawn
                    moves.append(('P', r, c, nr, nc, 'en_passant', cap_r))

    def _gen_knight_moves(self, r, c, p, white, moves):
        for dr, dc in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
            nr, nc = r + dr, c + dc
            if self.in_bounds(nr, nc):
                target = self.board[nr][nc]
                if target == '.' or self.is_enemy(target, white):
                    move_type = 'capture' if self.is_enemy(target, white) else 'normal'
                    moves.append(('N', r, c, nr, nc, move_type))

    def _gen_sliding_moves(self, r, c, p, white, moves, directions):
        for dr, dc in directions:
            for dist in range(1, 8):
                nr, nc = r + dr * dist, c + dc * dist
                if not self.in_bounds(nr, nc):
                    break
                target = self.board[nr][nc]
                if target == '.':
                    moves.append((p.upper(), r, c, nr, nc, 'normal'))
                elif self.is_enemy(target, white):
                    moves.append((p.upper(), r, c, nr, nc, 'capture'))
                    break
                else:
                    break

    def _gen_king_moves(self, r, c, p, white, moves):
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if self.in_bounds(nr, nc):
                    target = self.board[nr][nc]
                    if target == '.' or self.is_enemy(target, white):
                        move_type = 'capture' if self.is_enemy(target, white) else 'normal'
                        moves.append(('K', r, c, nr, nc, move_type))

        # Castling
        if white:
            if self.castling_rights['K']:
                if (self.board[7][5] == '.' and self.board[7][6] == '.'):
                    if (not self.is_square_attacked(7, 4, False) and
                        not self.is_square_attacked(7, 5, False) and
                        not self.is_square_attacked(7, 6, False)):
                        moves.append(('K', 7, 4, 7, 6, 'castle_king'))
            if self.castling_rights['Q']:
                if (self.board[7][3] == '.' and self.board[7][2] == '.' and self.board[7][1] == '.'):
                    if (not self.is_square_attacked(7, 4, False) and
                        not self.is_square_attacked(7, 3, False) and
                        not self.is_square_attacked(7, 2, False)):
                        moves.append(('K', 7, 4, 7, 2, 'castle_queen'))
        else:
            if self.castling_rights['k']:
                if (self.board[0][5] == '.' and self.board[0][6] == '.'):
                    if (not self.is_square_attacked(0, 4, True) and
                        not self.is_square_attacked(0, 5, True) and
                        not self.is_square_attacked(0, 6, True)):
                        moves.append(('K', 0, 4, 0, 6, 'castle_king'))
            if self.castling_rights['q']:
                if (self.board[0][3] == '.' and self.board[0][2] == '.' and self.board[0][1] == '.'):
                    if (not self.is_square_attacked(0, 4, True) and
                        not self.is_square_attacked(0, 3, True) and
                        not self.is_square_attacked(0, 2, True)):
                        moves.append(('K', 0, 4, 0, 2, 'castle_queen'))

    def generate_legal_moves(self, white=None):
        if white is None:
            white = self.white_to_move
        pseudo = self.generate_pseudo_legal_moves(white)
        legal = []
        for move in pseudo:
            test = self.copy()
            test.make_move(move)
            # After move, check if our king is in check
            if not test.in_check(white):
                legal.append(move)
        return legal

    def make_move(self, move):
        """Apply a move to the board in place."""
        piece, fr, fc, tr, tc, move_type = move[:6]
        promo_piece = move[6] if len(move) > 6 else None

        moving_piece = self.board[fr][fc]

        # Update en passant
        self.en_passant_sq = None

        # Reset halfmove clock on pawn move or capture
        if piece == 'P' or 'capture' in move_type:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1

        # Handle special moves
        if move_type == 'double_push':
            self.board[tr][tc] = moving_piece
            self.board[fr][fc] = '.'
            # Set en passant square
            ep_r = (fr + tr) // 2
            self.en_passant_sq = (ep_r, fc)

        elif move_type == 'en_passant':
            self.board[tr][tc] = moving_piece
            self.board[fr][fc] = '.'
            cap_r = move[6]  # row of captured pawn
            self.board[cap_r][tc] = '.'

        elif move_type == 'capture_en_passant':
            self.board[tr][tc] = moving_piece
            self.board[fr][fc] = '.'
            cap_r = move[6]
            self.board[cap_r][tc] = '.'

        elif move_type in ('promote', 'capture_promote'):
            self.board[tr][tc] = promo_piece
            self.board[fr][fc] = '.'

        elif move_type == 'castle_king':
            self.board[tr][tc] = moving_piece
            self.board[fr][fc] = '.'
            # Move rook
            rook_c = 7 if tc == 6 else 5
            rook_piece = self.board[fr][7] if tc == 6 else self.board[fr][5]
            dest_rook_c = 5 if tc == 6 else 3
            self.board[fr][dest_rook_c] = rook_piece
            self.board[fr][7 if tc == 6 else 5] = '.'

        elif move_type == 'castle_queen':
            self.board[tr][tc] = moving_piece
            self.board[fr][fc] = '.'
            rook_piece = self.board[fr][0]
            self.board[fr][3] = rook_piece
            self.board[fr][0] = '.'

        else:
            # Normal move or capture
            self.board[tr][tc] = moving_piece
            self.board[fr][fc] = '.'

        # Update castling rights
        if piece == 'K':
            if white_turn_from_piece(piece):
                self.castling_rights['K'] = False
                self.castling_rights['Q'] = False
            else:
                self.castling_rights['k'] = False
                self.castling_rights['q'] = False

        if piece == 'R':
            if fr == 7 and fc == 7:
                self.castling_rights['K'] = False
            elif fr == 7 and fc == 0:
                self.castling_rights['Q'] = False
            elif fr == 0 and fc == 7:
                self.castling_rights['k'] = False
            elif fr == 0 and fc == 0:
                self.castling_rights['q'] = False

        # If rook is captured, remove castling rights
        if tr == 0 and tc == 7:
            self.castling_rights['k'] = False
        elif tr == 0 and tc == 0:
            self.castling_rights['q'] = False
        elif tr == 7 and tc == 7:
            self.castling_rights['K'] = False
        elif tr == 7 and tc == 0:
            self.castling_rights['Q'] = False

        # Toggle turn
        self.white_to_move = not self.white_to_move
        if self.white_to_move:
            self.fullmove_number += 1

    def undo_move(self, move):
        """Undo a move (requires move log info)."""
        piece, fr, fc, tr, tc, move_type = move[:6]
        promo_piece = move[6] if len(move) > 6 else None

        # We need the captured piece info — stored in move log
        # This is a simplified undo; for AI we use copy instead
        pass

    def is_checkmate(self, white=None):
        if white is None:
            white = self.white_to_move
        return self.in_check(white) and len(self.generate_legal_moves(white)) == 0

    def is_stalemate(self, white=None):
        if white is None:
            white = self.white_to_move
        return not self.in_check(white) and len(self.generate_legal_moves(white)) == 0

    def is_draw(self):
        if self.halfmove_clock >= 100:
            return True
        if self.is_stalemate():
            return True
        # Insufficient material
        pieces = []
        for r in range(8):
            for c in range(8):
                p = self.board[r][c]
                if p != '.' and p.upper() != 'K':
                    pieces.append(p)
        if len(pieces) == 0:
            return True
        if len(pieces) == 1:
            if pieces[0].upper() in ('B', 'N'):
                return True
        return False

    def get_move_notation(self, move):
        piece, fr, fc, tr, tc, move_type = move[:6]
        promo_piece = move[6] if len(move) > 6 else None
        files = 'abcdefgh'
        ranks = '87654321'
        start = f"{files[fc]}{ranks[fr]}"
        end = f"{files[tc]}{ranks[tr]}"
        if piece == 'P':
            if 'capture' in move_type:
                return f"{files[fc]}x{end}"
            return end
        sym = piece.upper()
        notation = sym
        if 'capture' in move_type:
            notation += f"x{end}"
        else:
            notation += end
        if promo_piece:
            notation += f"={promo_piece.upper()}"
        return notation


def white_turn_from_piece(piece):
    return piece.isupper()


# ─── AI Engine ────────────────────────────────────────────────────────────────

class ChessAI:
    def __init__(self, depth=AI_DEPTH):
        self.depth = depth
        self.nodes_searched = 0
        self.transposition_table: Dict[str, float] = {}

    def evaluate(self, board: ChessBoard) -> float:
        """Greedy evaluation: material + positional bonuses."""
        score = 0
        for r in range(8):
            for c in range(8):
                p = board.board[r][c]
                if p == '.':
                    continue
                val = PIECE_VALUES.get(p, 0)
                # Positional bonus
                pst_key = p.upper()
                if pst_key in PST:
                    table = PST[pst_key]
                    if board.is_white(p):
                        pos_val = table[r * 8 + c]
                    else:
                        pos_val = table[(7 - r) * 8 + c]
                else:
                    pos_val = 0

                total = val + pos_val
                if board.is_white(p):
                    score += total
                else:
                    score -= total

        # Mobility bonus (simple)
        white_moves = len(board.generate_legal_moves(True))
        black_moves = len(board.generate_legal_moves(False))
        score += (white_moves - black_moves) * 5

        return score

    def order_moves(self, board: ChessBoard, moves):
        """Order moves for better alpha-beta pruning."""
        scored = []
        for move in moves:
            score = 0
            move_type = move[5]

            # Prioritize captures (MVV-LVA)
            if 'capture' in move_type:
                victim = board.board[move[3]][move[4]]
                attacker = board.board[move[1]][move[2]]
                score += 10 * PIECE_VALUES.get(victim, 0) - PIECE_VALUES.get(attacker, 0) + 10000

            # Prioritize promotions
            if 'promote' in move_type:
                score += 9000

            # Prioritize center control
            center_dist = abs(move[3] - 3.5) + abs(move[4] - 3.5)
            score -= center_dist * 10

            scored.append((score, move))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored]

    def minimax(self, board: ChessBoard, depth: int, alpha: float, beta: float,
                maximizing: bool) -> float:
        """Minimax with alpha-beta pruning."""
        self.nodes_searched += 1

        if depth == 0:
            return self.evaluate(board)

        moves = board.generate_legal_moves()

        if len(moves) == 0:
            if board.in_check(board.white_to_move):
                # Checkmate — worse for the side being mated
                return -99999 + (self.depth - depth) if maximizing else 99999 - (self.depth - depth)
            return 0  # Stalemate

        moves = self.order_moves(board, moves)

        if maximizing:
            max_eval = -999999
            for move in moves:
                child = board.copy()
                child.make_move(move)
                eval_score = self.minimax(child, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta cutoff
            return max_eval
        else:
            min_eval = 999999
            for move in moves:
                child = board.copy()
                child.make_move(move)
                eval_score = self.minimax(child, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha cutoff
            return min_eval

    def get_best_move(self, board: ChessBoard) -> Optional[Tuple]:
        """Get the best move for the current position."""
        self.nodes_searched = 0
        maximizing = board.white_to_move
        moves = board.generate_legal_moves()

        if not moves:
            return None

        moves = self.order_moves(board, moves)
        best_move = moves[0]
        best_score = -999999 if maximizing else 999999

        for move in moves:
            child = board.copy()
            child.make_move(move)
            score = self.minimax(child, self.depth - 1, -999999, 999999, not maximizing)

            if maximizing:
                if score > best_score:
                    best_score = score
                    best_move = move
            else:
                if score < best_score:
                    best_score = score
                    best_move = move

        print(f"AI searched {self.nodes_searched} nodes, score: {best_score}")
        return best_move


# ─── Pygame GUI ───────────────────────────────────────────────────────────────

class ChessGUI:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Chess Master AI")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        # Fonts — try multiple names to find one that supports chess symbols
        piece_font_name = None
        for name in ['dejavusans', 'arialunicodems', 'applesymbols', 'segoeuiymbol', 'helvetica']:
            if name in [f.lower() for f in pygame.font.get_fonts()]:
                piece_font_name = name
                break
        self.piece_font = pygame.font.SysFont(piece_font_name, 58) if piece_font_name else pygame.font.Font(None, 58)
        self.small_font = pygame.font.SysFont('arial', 16)
        self.medium_font = pygame.font.SysFont('arial', 22, bold=True)
        self.large_font = pygame.font.SysFont('arial', 28, bold=True)
        self.title_font = pygame.font.SysFont('georgia', 32, bold=True)

        # Game state
        self.board = ChessBoard()
        self.ai = ChessAI(depth=AI_DEPTH)
        self.selected_sq: Optional[Tuple[int, int]] = None
        self.legal_moves_for_selected: List = []
        self.last_move: Optional[Tuple] = None
        self.game_over = False
        self.game_result = ""
        self.ai_thinking = False
        self.player_white = True  # Player plays white by default
        self.move_history: List = []  # stores (notation, move_tuple) pairs
        self.dragging = False
        self.drag_piece = None
        self.drag_moved = False
        self.drag_pos = (0, 0)
        self.drag_start = None

        # Promotion dialog
        self.show_promotion = False
        self.promotion_moves = []
        self.promotion_row = 0

        # Generate unique board noise pattern for texture
        self.sq_texture = self._gen_sq_texture()

    def _gen_sq_texture(self):
        """Generate subtle texture offsets for each square for a hand-crafted feel."""
        import random
        random.seed(42)
        tex = {}
        for r in range(8):
            for c in range(8):
                is_light = (r + c) % 2 == 0
                base = LIGHT_SQ if is_light else DARK_SQ
                # Add slight per-pixel variation via a small surface
                surf = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE))
                for y in range(0, SQUARE_SIZE, 3):
                    for x in range(0, SQUARE_SIZE, 3):
                        variation = random.randint(-8, 8)
                        color = tuple(max(0, min(255, v + variation)) for v in base)
                        surf.fill(color, (x, y, 3, 3))
                tex[(r, c)] = surf
        return tex

    def draw_board(self):
        """Draw the chessboard with textured squares."""
        for r in range(8):
            for c in range(8):
                x = c * SQUARE_SIZE
                y = r * SQUARE_SIZE + 40  # top bar offset

                # Draw textured square
                self.screen.blit(self.sq_texture[(r, c)], (x, y))

        # Highlight squares
        self._draw_highlights()

    def _draw_highlights(self):
        # Last move highlights
        if self.last_move:
            piece, fr, fc, tr, tc, move_type = self.last_move[:6]
            for (r, c) in [(fr, fc), (tr, tc)]:
                surf = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                surf.fill(HIGHLIGHT_LAST)
                self.screen.blit(surf, (c * SQUARE_SIZE, r * SQUARE_SIZE + 40))

        # Check highlight
        if self.board.in_check(self.board.white_to_move):
            king = self.board.find_king(self.board.white_to_move)
            if king:
                surf = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                surf.fill(CHECK_COLOR)
                self.screen.blit(surf, (king[1] * SQUARE_SIZE, king[0] * SQUARE_SIZE + 40))

        # Selected square highlight
        if self.selected_sq:
            r, c = self.selected_sq
            surf = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            surf.fill(HIGHLIGHT_SRC)
            self.screen.blit(surf, (c * SQUARE_SIZE, r * SQUARE_SIZE + 40))

        # Legal move highlights
        for move in self.legal_moves_for_selected:
            tr, tc = move[3], move[4]
            surf = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            piece_at = self.board.board[tr][tc]
            if piece_at != '.' or move[5] == 'en_passant':
                # Capture: ring
                pygame.draw.circle(surf, (105, 247, 152, 140),
                                   (SQUARE_SIZE // 2, SQUARE_SIZE // 2), SQUARE_SIZE // 2 - 4, 4)
            else:
                # Move: dot
                pygame.draw.circle(surf, (105, 247, 152, 160),
                                   (SQUARE_SIZE // 2, SQUARE_SIZE // 2), 12)
            self.screen.blit(surf, (tc * SQUARE_SIZE, tr * SQUARE_SIZE + 40))

    def draw_pieces(self):
        """Draw chess pieces as Unicode symbols."""
        for r in range(8):
            for c in range(8):
                p = self.board.board[r][c]
                if p == '.':
                    continue

                # Skip piece being dragged
                if self.dragging and self.drag_start == (r, c):
                    continue

                x = c * SQUARE_SIZE + SQUARE_SIZE // 2
                y = r * SQUARE_SIZE + 40 + SQUARE_SIZE // 2

                symbol = UNICODE_PIECES.get(p, '?')

                # Draw piece with shadow for depth
                shadow_surf = self.piece_font.render(symbol, True, (30, 28, 25))
                shadow_rect = shadow_surf.get_rect(center=(x + 2, y + 3))
                self.screen.blit(shadow_surf, shadow_rect)

                # Draw piece
                piece_color = (255, 252, 245) if self.board.is_white(p) else (45, 42, 38)
                piece_surf = self.piece_font.render(symbol, True, piece_color)
                piece_rect = piece_surf.get_rect(center=(x, y))
                self.screen.blit(piece_surf, piece_rect)

        # Draw dragged piece
        if self.dragging and self.drag_piece:
            symbol = UNICODE_PIECES.get(self.drag_piece, '?')
            piece_color = (255, 252, 245) if self.board.is_white(self.drag_piece) else (45, 42, 38)
            shadow_surf = self.piece_font.render(symbol, True, (30, 28, 25))
            shadow_rect = shadow_surf.get_rect(center=(self.drag_pos[0] + 2, self.drag_pos[1] + 3))
            self.screen.blit(shadow_surf, shadow_rect)

            piece_surf = self.piece_font.render(symbol, True, piece_color)
            piece_rect = piece_surf.get_rect(center=self.drag_pos)
            self.screen.blit(piece_surf, piece_rect)

    def draw_sidebar(self):
        """Draw the right sidebar with info and controls."""
        sidebar_x = BOARD_PX
        pygame.draw.rect(self.screen, SIDEBAR_BG, (sidebar_x, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT))

        # Title
        title = self.title_font.render("Chess Master AI", True, ACCENT_COLOR)
        self.screen.blit(title, (sidebar_x + 15, 15))

        # Separator line
        pygame.draw.line(self.screen, ACCENT_COLOR, (sidebar_x + 15, 55), (sidebar_x + SIDEBAR_WIDTH - 15, 55), 2)

        # Turn indicator
        y = 70
        turn_text = "White's Turn" if self.board.white_to_move else "Black's Turn"
        turn_color = (255, 252, 245) if self.board.white_to_move else (180, 175, 170)
        turn_surf = self.medium_font.render(turn_text, True, turn_color)
        self.screen.blit(turn_surf, (sidebar_x + 15, y))

        if self.board.in_check(self.board.white_to_move):
            check_surf = self.small_font.render("  CHECK!", True, (235, 64, 52))
            self.screen.blit(check_surf, (sidebar_x + 15 + turn_surf.get_width(), y + 4))

        # AI thinking indicator
        y += 35
        if self.ai_thinking:
            think_surf = self.small_font.render("AI is thinking...", True, ACCENT_COLOR)
            self.screen.blit(think_surf, (sidebar_x + 15, y))
        elif self.game_over:
            result_surf = self.large_font.render(self.game_result, True, ACCENT_COLOR)
            self.screen.blit(result_surf, (sidebar_x + 15, y))

        # Move history
        y += 50
        hist_label = self.small_font.render("Moves:", True, TEXT_COLOR)
        self.screen.blit(hist_label, (sidebar_x + 15, y))
        y += 22

        # Show last N moves
        max_visible = 16
        visible_moves = self.move_history[-max_visible:] if self.move_history else []
        for i, entry in enumerate(visible_moves):
            notation = entry[0] if isinstance(entry, tuple) else entry
            move_num = len(self.move_history) - len(visible_moves) + i + 1
            if i % 2 == 0:
                text = f"{(move_num // 2) + 1}. {notation}"
            else:
                text = f"   {notation}"
            move_surf = self.small_font.render(text, True, TEXT_COLOR)
            self.screen.blit(move_surf, (sidebar_x + 15, y + i * 18))

        # Buttons
        btn_y = WINDOW_HEIGHT - 140
        self._draw_button(sidebar_x + 15, btn_y, SIDEBAR_WIDTH - 30, 38, "New Game", "new_game")
        self._draw_button(sidebar_x + 15, btn_y + 48, SIDEBAR_WIDTH - 30, 38, "Undo Move", "undo")

        # Difficulty selector
        diff_y = btn_y + 110
        diff_label = self.small_font.render("AI Depth:", True, TEXT_COLOR)
        self.screen.blit(diff_label, (sidebar_x + 15, diff_y))

        for i, depth in enumerate([2, 3, 4]):
            bx = sidebar_x + 15 + i * 65
            by = diff_y + 20
            active = self.ai.depth == depth
            color = ACCENT_COLOR if active else BUTTON_COLOR
            pygame.draw.rect(self.screen, color, (bx, by, 55, 28), border_radius=5)
            dtext = self.small_font.render(str(depth), True, BUTTON_TEXT)
            self.screen.blit(dtext, (bx + 22 - dtext.get_width() // 2, by + 5))

    def _draw_button(self, x, y, w, h, text, action):
        """Draw a clickable button."""
        mouse_pos = pygame.mouse.get_pos()
        hover = x <= mouse_pos[0] <= x + w and y <= mouse_pos[1] <= y + h
        color = BUTTON_HOVER if hover else BUTTON_COLOR
        pygame.draw.rect(self.screen, color, (x, y, w, h), border_radius=6)
        text_surf = self.small_font.render(text, True, BUTTON_TEXT)
        text_rect = text_surf.get_rect(center=(x + w // 2, y + h // 2))
        self.screen.blit(text_surf, text_rect)

    def draw_top_bar(self):
        """Draw the top info bar."""
        pygame.draw.rect(self.screen, BG_COLOR, (0, 0, WINDOW_WIDTH, 40))
        pygame.draw.line(self.screen, ACCENT_COLOR, (0, 40), (WINDOW_WIDTH, 40), 1)

        # File and rank labels
        files = 'abcdefgh'
        ranks = '87654321'
        for i in range(8):
            # File labels (bottom)
            label = self.small_font.render(files[i], True, (140, 135, 128))
            self.screen.blit(label, (i * SQUARE_SIZE + SQUARE_SIZE // 2 - 4, BOARD_PX + 42))

        for i in range(8):
            # Rank labels (left)
            label = self.small_font.render(ranks[i], True, (140, 135, 128))
            self.screen.blit(label, (3, i * SQUARE_SIZE + 44))

    def draw_promotion_dialog(self):
        """Draw piece selection for pawn promotion."""
        if not self.show_promotion:
            return

        # Semi-transparent overlay
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        self.screen.blit(overlay, (0, 0))

        # Dialog box
        dialog_w = 360
        dialog_h = 100
        dialog_x = (BOARD_PX - dialog_w) // 2
        dialog_y = (WINDOW_HEIGHT - dialog_h) // 2

        pygame.draw.rect(self.screen, SIDEBAR_BG, (dialog_x, dialog_y, dialog_w, dialog_h), border_radius=10)
        pygame.draw.rect(self.screen, ACCENT_COLOR, (dialog_x, dialog_y, dialog_w, dialog_h), 2, border_radius=10)

        label = self.medium_font.render("Choose promotion:", True, TEXT_COLOR)
        self.screen.blit(label, (dialog_x + 10, dialog_y + 8))

        white = self.board.white_to_move
        pieces = ['Q', 'R', 'B', 'N'] if white else ['q', 'r', 'b', 'n']

        for i, p in enumerate(pieces):
            px = dialog_x + 30 + i * 85
            py = dialog_y + 45
            rect = pygame.Rect(px, py, 70, 45)
            mouse_pos = pygame.mouse.get_pos()
            hover = rect.collidepoint(mouse_pos)
            color = ACCENT_COLOR if hover else BUTTON_COLOR
            pygame.draw.rect(self.screen, color, rect, border_radius=6)

            symbol = UNICODE_PIECES.get(p, '?')
            sym_surf = self.piece_font.render(symbol, True, (255, 252, 245) if white else (45, 42, 38))
            self.screen.blit(sym_surf, sym_surf.get_rect(center=rect.center))

    def draw_game_over(self):
        """Draw game over overlay."""
        if not self.game_over:
            return

        overlay = pygame.Surface((BOARD_PX, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 100))
        self.screen.blit(overlay, (0, 40))

        # Result box
        box_w = 300
        box_h = 120
        box_x = (BOARD_PX - box_w) // 2
        box_y = (WINDOW_HEIGHT - box_h) // 2

        pygame.draw.rect(self.screen, (44, 40, 37), (box_x, box_y, box_w, box_h), border_radius=12)
        pygame.draw.rect(self.screen, ACCENT_COLOR, (box_x, box_y, box_w, box_h), 2, border_radius=12)

        title = self.title_font.render(self.game_result, True, ACCENT_COLOR)
        self.screen.blit(title, title.get_rect(center=(box_x + box_w // 2, box_y + 40)))

        hint = self.small_font.render("Click 'New Game' to play again", True, TEXT_COLOR)
        self.screen.blit(hint, hint.get_rect(center=(box_x + box_w // 2, box_y + 85)))

    def board_to_screen(self, pos):
        """Convert screen position to board coordinates."""
        x, y = pos
        if y < 40:
            return None
        c = x // SQUARE_SIZE
        r = (y - 40) // SQUARE_SIZE
        if 0 <= r < 8 and 0 <= c < 8:
            return (r, c)
        return None

    def handle_click(self, pos, button):
        """Handle mouse click."""
        if self.game_over:
            return

        if self.show_promotion:
            self._handle_promotion_click(pos)
            return

        # Check sidebar button clicks
        sidebar_x = BOARD_PX
        mx, my = pos

        # New Game button
        btn_y = WINDOW_HEIGHT - 140
        if sidebar_x + 15 <= mx <= sidebar_x + SIDEBAR_WIDTH - 15:
            if btn_y <= my <= btn_y + 38:
                self.new_game()
                return
            if btn_y + 48 <= my <= btn_y + 86:
                self.undo_move()
                return

            # Depth buttons
            diff_y = btn_y + 130
            for i, depth in enumerate([2, 3, 4]):
                bx = sidebar_x + 15 + i * 65
                by = diff_y
                if bx <= mx <= bx + 55 and by <= my <= by + 28:
                    self.ai.depth = depth
                    return

        # Board click
        sq = self.board_to_screen(pos)
        if sq is None:
            return

        r, c = sq

        # If AI is thinking, ignore clicks
        if self.ai_thinking:
            return

        # Is it the player's turn?
        is_player_turn = (self.player_white and self.board.white_to_move) or \
                         (not self.player_white and not self.board.white_to_move)
        if not is_player_turn:
            return

        if button == 1:  # Left click
            if self.selected_sq:
                # Try to make a move
                move = self._find_matching_move(r, c)
                if move:
                    self._execute_move(move)
                    return

            # Select a piece
            piece = self.board.board[r][c]
            if piece != '.':
                is_white_piece = self.board.is_white(piece)
                if (self.player_white and is_white_piece) or \
                   (not self.player_white and not is_white_piece):
                    self.selected_sq = (r, c)
                    self.legal_moves_for_selected = [
                        m for m in self.board.generate_legal_moves()
                        if m[1] == r and m[2] == c
                    ]
                else:
                    # Clicked enemy piece — try capture
                    move = self._find_matching_move(r, c)
                    if move:
                        self._execute_move(move)
            else:
                self.selected_sq = None
                self.legal_moves_for_selected = []

        elif button == 3:  # Right click — deselect
            self.selected_sq = None
            self.legal_moves_for_selected = []

    def _find_matching_move(self, tr, tc):
        """Find a legal move matching the target square."""
        for move in self.legal_moves_for_selected:
            if move[3] == tr and move[4] == tc:
                return move
        return None

    def _execute_move(self, move):
        """Execute a player move."""
        notation = self.board.get_move_notation(move)

        # Check if this is a promotion
        if move[5] in ('promote', 'capture_promote'):
            self.show_promotion = True
            self.promotion_moves = [m for m in self.legal_moves_for_selected
                                    if m[3] == move[3] and m[4] == move[4] and 'promote' in m[5]]
            self.promotion_row = move[3]
            return

        self.board.make_move(move)
        self.last_move = move
        self.move_history.append((notation, move))
        self.selected_sq = None
        self.legal_moves_for_selected = []

        # Check game state
        self._check_game_state()

        if not self.game_over:
            # AI turn
            self.ai_thinking = True

    def _handle_promotion_click(self, pos):
        """Handle click on promotion dialog."""
        mx, my = pos
        dialog_w = 360
        dialog_x = (BOARD_PX - dialog_w) // 2
        dialog_y = (WINDOW_HEIGHT - 100) // 2

        white = self.board.white_to_move
        pieces = ['Q', 'R', 'B', 'N'] if white else ['q', 'r', 'b', 'n']

        for i, p in enumerate(pieces):
            px = dialog_x + 30 + i * 85
            py = dialog_y + 45
            rect = pygame.Rect(px, py, 70, 45)
            if rect.collidepoint(pos):
                # Find the matching move
                for move in self.promotion_moves:
                    if move[6] == p:
                        notation = self.board.get_move_notation(move)
                        self.board.make_move(move)
                        self.last_move = move
                        self.move_history.append((notation, move))
                        break

                self.show_promotion = False
                self.promotion_moves = []
                self.selected_sq = None
                self.legal_moves_for_selected = []
                self._check_game_state()

                if not self.game_over:
                    self.ai_thinking = True
                break

    def _check_game_state(self):
        """Check for checkmate, stalemate, or draw."""
        if self.board.is_checkmate():
            winner = "White" if not self.board.white_to_move else "Black"
            self.game_result = f"{winner} wins!"
            self.game_over = True
        elif self.board.is_stalemate():
            self.game_result = "Stalemate"
            self.game_over = True
        elif self.board.is_draw():
            self.game_result = "Draw"
            self.game_over = True

    def ai_move(self):
        """Make the AI's move."""
        if self.game_over:
            return

        move = self.ai.get_best_move(self.board)
        if move is None:
            return

        notation = self.board.get_move_notation(move)
        self.board.make_move(move)
        self.last_move = move
        self.move_history.append((notation, move))
        self.ai_thinking = False

        self._check_game_state()

    def undo_move(self):
        """Undo the last two moves (player + AI)."""
        if len(self.move_history) < 2 or self.ai_thinking or self.game_over:
            return

        # Remove last two moves (player + AI response)
        self.move_history = self.move_history[:-2]

        # Replay all remaining moves from scratch
        self.board = ChessBoard()
        self.last_move = None
        for entry in self.move_history:
            if isinstance(entry, tuple):
                _, move = entry
            else:
                continue
            self.board.make_move(move)
            self.last_move = move

        self.selected_sq = None
        self.legal_moves_for_selected = []
        self.ai_thinking = False

    def new_game(self):
        """Start a new game."""
        self.board = ChessBoard()
        self.selected_sq = None
        self.legal_moves_for_selected = []
        self.last_move = None
        self.game_over = False
        self.game_result = ""
        self.ai_thinking = False
        self.move_history = []
        self.show_promotion = False
        self.promotion_moves = []
        self.dragging = False
        self.drag_piece = None
        self.drag_moved = False

    def _handle_sidebar_click(self, pos):
        """Handle clicks on sidebar buttons and depth selector."""
        sidebar_x = BOARD_PX
        mx, my = pos
        if mx < sidebar_x:
            return False

        btn_y = WINDOW_HEIGHT - 140
        if sidebar_x + 15 <= mx <= sidebar_x + SIDEBAR_WIDTH - 15:
            if btn_y <= my <= btn_y + 38:
                self.new_game()
                return True
            if btn_y + 48 <= my <= btn_y + 86:
                self.undo_move()
                return True

            diff_y = btn_y + 130
            for i, depth in enumerate([2, 3, 4]):
                bx = sidebar_x + 15 + i * 65
                by = diff_y
                if bx <= mx <= bx + 55 and by <= my <= by + 28:
                    self.ai.depth = depth
                    return True
        return False

    def _is_player_turn(self):
        """Check if it's the human player's turn to move."""
        return (self.player_white and self.board.white_to_move) or \
               (not self.player_white and not self.board.white_to_move)

    def _can_interact(self):
        """Check if the player can interact with the board."""
        return not self.game_over and not self.ai_thinking and not self.show_promotion and self._is_player_turn()

    def run(self):
        """Main game loop."""
        running = True
        self.drag_moved = False

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        # Check sidebar first
                        if self._handle_sidebar_click(event.pos):
                            continue

                        # Handle promotion dialog
                        if self.show_promotion:
                            self._handle_promotion_click(event.pos)
                            continue

                        if not self._can_interact():
                            continue

                        # Check if clicking on a friendly piece → start drag + select
                        sq = self.board_to_screen(event.pos)
                        if sq is None:
                            continue
                        r, c = sq
                        piece = self.board.board[r][c]

                        if piece != '.':
                            is_white_piece = self.board.is_white(piece)
                            friendly = (self.player_white and is_white_piece) or \
                                       (not self.player_white and not is_white_piece)
                            if friendly:
                                # Select piece and prepare for drag
                                self.selected_sq = (r, c)
                                self.legal_moves_for_selected = [
                                    m for m in self.board.generate_legal_moves()
                                    if m[1] == r and m[2] == c
                                ]
                                self.dragging = True
                                self.drag_piece = piece
                                self.drag_start = (r, c)
                                self.drag_pos = event.pos
                                self.drag_moved = False
                            else:
                                # Clicked enemy piece — try capture if we have a selection
                                if self.selected_sq:
                                    move = self._find_matching_move(r, c)
                                    if move:
                                        self._execute_move(move)
                        else:
                            # Clicked empty square — try to move there if we have a selection
                            if self.selected_sq:
                                move = self._find_matching_move(r, c)
                                if move:
                                    self._execute_move(move)
                                else:
                                    self.selected_sq = None
                                    self.legal_moves_for_selected = []
                            else:
                                self.selected_sq = None
                                self.legal_moves_for_selected = []

                    elif event.button == 3:
                        # Right click — deselect
                        self.selected_sq = None
                        self.legal_moves_for_selected = []

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1 and self.dragging:
                        self.dragging = False
                        sq = self.board_to_screen(event.pos)

                        if sq and self.drag_start and sq != self.drag_start and self.drag_moved:
                            # Drag released on different square — try to move
                            move = self._find_matching_move(sq[0], sq[1])
                            if move:
                                self._execute_move(move)
                                # Clear selection since drag-move handled it
                                self.selected_sq = None
                                self.legal_moves_for_selected = []

                        self.drag_piece = None
                        self.drag_start = None
                        self.drag_moved = False

                elif event.type == pygame.MOUSEMOTION:
                    if self.dragging:
                        self.drag_pos = event.pos
                        self.drag_moved = True

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_n:
                        self.new_game()
                    elif event.key == pygame.K_z:
                        self.undo_move()
                    elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                        self.ai.depth = min(5, self.ai.depth + 1)
                    elif event.key == pygame.K_MINUS:
                        self.ai.depth = max(1, self.ai.depth - 1)

            # AI turn
            if self.ai_thinking and not self.game_over:
                self.ai_move()

            # Draw everything
            self.screen.fill(BG_COLOR)
            self.draw_board()
            self.draw_pieces()
            self.draw_top_bar()
            self.draw_sidebar()
            self.draw_promotion_dialog()
            self.draw_game_over()

            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()


if __name__ == '__main__':
    game = ChessGUI()
    game.run()
