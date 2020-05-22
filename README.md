# pyludo
Artificial intelligence algorithms made for the Ludo game. 
Two main algorithms are made 

## State
The state of the pyludo game is a description of each token's position

## Reduced state 
The reduces state takes the pyldo game's state and makes it reduced by instead representing each token's position by a number from 0-8.

## Q-Learning
Uses the reduced state space.
### Token-based Q-Learning
Token-based Q-Learning algorithm choose which token to move based on the reduced state space.

### Action-based Q-Learning
Action-based Q-Learning algortihm choose which game action to make based on the reduced state space. The game actions are:
- **Move out of spawn:** Moves a token out from home. 
- **Move into goal:** Moves a token into the goal
- **Send opponent home:** Moves a token into the opponent's token so the opponents token is send home
- **Send self home:** Moves a token so it is send home
- **Move token:** Moves one token

The actions chosen are mapped onto which token can make the move.

## Simple GA
Learns the genes of the chromosomes which weights the actions:
- **Move out of spawn:** Moves a token out from home. 
- **Move into goal:** Moves a token into the goal
- **Send opponent home:** Moves a token into the opponent's token so the opponents token is send home
- **Send self home:** Moves a token so it is send home
