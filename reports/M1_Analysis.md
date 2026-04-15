# satellite-vision-project
# M1 Analysis

## 1. Backpropagation Summary

Backpropagation is the process used to train neural networks by updating weights to minimize loss.

Steps:
1. Perform forward pass to compute predictions.
2. Compute loss using a loss function.
3. Calculate gradients of loss with respect to output.
4. Propagate gradients backward through layers using chain rule.
5. Update weights using gradient descent.

---

## 2. Why FFN is inefficient for images

Feedforward Neural Networks are not suitable for image data because:

- They flatten images → lose spatial information
- Huge number of parameters
- No feature sharing

---

## 3. Why CNN is better

Convolutional Neural Networks:

- Preserve spatial structure
- Use filters to detect features
- Parameter sharing reduces complexity
- Better performance on images