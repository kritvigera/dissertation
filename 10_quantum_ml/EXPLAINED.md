# Stage 10, said simply

## What this stage does

Asks whether a small quantum computer program can learn the judge's job —
real column layout or broken one? — and answers by racing it against
classical programs given *exactly* the same information.

## How a quantum circuit classifies anything

Four measurements describe each layout (how uneven in x, how uneven in y,
how many interior columns, how regular overall). Each number becomes the
rotation angle of one qubit — turn the dial by that much. The qubits are
then entangled and rotated by trainable amounts, and one qubit is measured
at the end: closer to +1 means "real", closer to −1 means "broken".
Training nudges the trainable rotations exactly the way any neural network
is trained. All of it runs on a simulator — an ordinary computer imitating
a perfect quantum one — because four qubits is small enough to imitate and
today's real hardware would only add noise.

## The one rule that makes the race fair

Never compare a pony against a freight train and publish the winner. The
fair ladder has a rung for every doubt:

* a **matched** classical model — logistic regression on the same four
  numbers and the same few hundred examples. This is the pony-sized
  classical control;
* classical boosting on those same four numbers;
* classical boosting on all six numbers, same examples — what capacity
  alone buys;
* classical boosting on all six numbers and *all twelve thousand plans* —
  what data alone buys.

## What happened

The quantum circuits scored 0.78–0.79 (out of 1). The matched classical
control scored 0.78. Dead level. Boosting with the same four numbers
actually did *worse* (0.75). Give the classical side its full six features
and the full corpus, and it walks away (0.88).

Two sentences carry the whole story. **At equal information, quantum keeps
up.** **Everything more information buys is still classical.** And one
number keeps everyone humble: the fanciest circuit took fourteen minutes of
simulation to reach the score logistic regression reached in a blink.

## Why the thesis keeps its old failure on display

An earlier version of this comparison, run on the flawed ground truth
(columns at every crossing), showed the quantum ensemble *matching* a
strong classical model — a result that evaporated the day the ground truth
was fixed. It stays in the write-up as a worked example of how easily
quantum-versus-classical comparisons flatter, and why this stage's real
contribution is its controls, not its circuits.
