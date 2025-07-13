function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function sigmoidDerivative(x) {
  const s = sigmoid(x);
  return s * (1 - s);
}

let input = 0.5;
let target = 1.0;
let lr = 0.1;

let w1 = Math.random();
let b1 = Math.random();

let w2 = Math.random();
let b2 = Math.random();

function forward(x) {
  const z1 = w1 * x + b1;
  const h = sigmoid(z1);
  const z2 = w2 * h + b2;
  const output = sigmoid(z2);
  return { z1, h, z2, output };
}

for (let i = 0; i < 1000; i++) {
  const { z1, h, z2, output } = forward(input);
  const error = output - target;

  const dOutput = error * sigmoidDerivative(z2);
  const dHidden = dOutput * w2 * sigmoidDerivative(z1);

  w2 -= lr * dOutput * h;
  b2 -= lr * dOutput;

  w1 -= lr * dHidden * input;
  b1 -= lr * dHidden;
}

const { output: finalOutput } = forward(input);
console.log(`finalOutput: ${finalOutput.toFixed(4)}`);
console.log(`Weights: Weight 1: ${w1.toFixed(4)} Bias 1: ${b1.toFixed(4)}`);
console.log(`Biases: Weight 2: ${w2.toFixed(4)} Bias 2: ${b2.toFixed(4)}`);
