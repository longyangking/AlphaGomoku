from enum import Enum 

# ChessBoard Info
ChessBoardHeight = 20
ChessBoardWidth = 20

# Role Info
Role = Enum('Role','Player Computer')

# Chess Info
Chess = Enum('Chess','PutByPlayer PutByComputer NoChess')

# Victory Info
Victory = Enum('Victory','PlayerWin ComputerWin NoOneWin')