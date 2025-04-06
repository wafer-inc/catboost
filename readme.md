# Catboost inference

There are some [Catboost Rust](https://github.com/jlloh/catboost-rs) crates, but they're based on bindings to the C++ API and handle both training and inference, which makes them highly complicated to build and use. This is a simple library that just handles inference.

To use, save your catboost classifier to JSON, like so (python):

```python
classifier.save_model(
    "my-model",
    format="json",
)
```

Then use it from Rust like so:

```rust
use catboost::Catboost;
use std::path::Path;

let model = CatBoost::load(Path::new("my-model.json")).unwrap();
let test_features: Vec<f32> = vec![0.1276993, 0.9918129, 0.16597846, 0.98612934];
let probability = model.predict(&test_features).unwrap();
```

Note: this library does not currently support categorical features. (Only float features are supported.) But categorical feature support would probably be pretty simple to add. [Leave an issue](https://github.com/wafer-inc/catboost/issues/new) if that's something you're interested in.
