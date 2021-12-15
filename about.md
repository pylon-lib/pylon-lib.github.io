---
title: About
layout: page
description: About
bodyClass: page-about
---

  Deep learning excels at learning low-level features from unstructured data but struggles with learning high-level procedural knowledge which can often times be symbolically and succinctly represented.
  Pylon is an open-source package that build on PyTorch to reconcile imperatively trained networks with procedurally specified knowledge.
  It provides a framework where users can programmatically specify constraints as Python functions and compile them into a differentiable loss function, learning predictive models that fit the data _whilst_ satisfying the specified constraints.
  Pylon includes both exact as well as approximate compilers, employing fuzzy logic, sampling methods or MAP estimates to ensure scalability to problems where exactness might be infeasible.
  Sampling compilers provide a particularly unique opportunity for users to incorporate external resources, which are inherently indifferentiable, or express constraints that might otherwise be arduous or impossible to specify in logic.
  Crucially, a guiding principle in designing Pylon has been the ease with which any existing deep learning code fragment, using either the CPU or GPU, could be extended to handle constraints in only two lines of code: the first defining the loss function in terms of a custom constraint, and the second computing the loss.
