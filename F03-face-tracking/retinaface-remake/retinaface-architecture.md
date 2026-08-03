Input image
    │
    ▼
ResNet-50 backbone
    │
    ├── C3: stride 8
    ├── C4: stride 16
    └── C5: stride 32
          │
          ▼
Feature Pyramid Network
    │
    ├── P3: small faces
    ├── P4: medium faces
    └── P5: large faces
          │
          ▼
Context modules / SSH blocks
    │
    ├── classification head
    ├── box regression head
    └── landmark regression head



FPN :
C5 ────────────────► P5
 │                    │
 │ upsample            │
 ▼                    ▼
C4 + upsample(P5) ───► P4
 │                    │
 │ upsample            │
 ▼                    ▼
C3 + upsample(P4) ───► P3