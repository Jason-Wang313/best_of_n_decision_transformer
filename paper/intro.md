# Introduction

Decision Transformers make return a prompt. That turns inference into an optimization surface: ask for a larger return, sample several candidate continuations, and keep the best one under an internal score. This can be useful when the score remains aligned with real utility. It can also create a return-conditioning fantasy when the prompt asks for returns beyond the offline dataset support.

This paper studies that mechanism in a controlled setting. The goal is not benchmark dominance; it is a precise diagnostic for when higher selected proxy return should not be mistaken for higher utility.

