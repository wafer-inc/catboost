//! # CatBoost Inference
//!
//! This library allows you to load and perform inference with CatBoost models.
//!
//! To use this library, you need to have a CatBoost model saved in JSON format. If using the python catboost library, you can save the model to JSON like this:
//!
//! ```python
//! classifier.save_model(
//!     model_filename,
//!     format="json",
//! )
//! ```
//!
//! Then use the [`CatBoost`] struct to load the model and perform inference.
//!
//! Note that categorical features are not supported at this time. (Only float features are supported.)

#![allow(unused)]
#![deny(missing_docs)]

use serde::Deserialize;
use std::{fs::File, io::BufReader, path::Path};
use thiserror::Error;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

/// The main struct for the CatBoost model.
/// This struct contains the model parameters and methods for loading and performing inference.
///
/// You most likely want to use [`CatBoost::load`] or [`CatBoost::try_from_json`] to create an instance of this struct.
///
/// Then, use [`CatBoost::predict`] or [`CatBoost::predict_raw`] to perform inference.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct CatBoost {
    features_info: Features,
    oblivious_trees: Vec<ObliviousTree>,
    scale_and_bias: (f32, Vec<f32>),
}

impl CatBoost {
    /// Loads a CatBoost model from a JSON file.
    ///
    /// ```rust
    /// # use catboost::CatBoost;
    /// # use std::path::Path;
    /// let model = CatBoost::load(Path::new("models/test/tiny-binary-catboost.json")).unwrap();
    /// ```
    pub fn load(path: &Path) -> Result<Self, std::io::Error> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let model: CatBoost = serde_json::from_reader(reader)?;
        Ok(model)
    }

    /// Loads a CatBoost model from a JSON string.
    ///
    /// ```rust
    /// use catboost::CatBoost;
    /// let model_str = include_str!("../models/test/tiny-binary-catboost.json");
    /// let model = CatBoost::try_from_json(model_str).unwrap();
    /// ```
    pub fn try_from_json(model_str: &str) -> Result<Self, serde_json::Error> {
        let model: CatBoost = serde_json::from_str(model_str)?;
        Ok(model)
    }

    fn num_features(&self) -> usize {
        self.features_info
            .float_features
            .iter()
            .map(|f| f.flat_feature_index)
            .max()
            .map_or(0, |m| m + 1)
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
struct Features {
    float_features: Vec<FloatFeature>,
}

/// A float feature.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct FloatFeature {
    /// feature index among only float features, zero-based indexation.
    feature_index: usize,
    ///  feature index in pool, zero-based indexation
    flat_feature_index: usize,
    /// The borders of the feature. These are all the places where this feature is split.
    borders: Vec<f32>,
    /// Whether the feature has NaN values.
    has_nans: bool,
    /// How to handle NaN values.
    nan_value_treatment: NanValueTreatment,
}

/// How to handle NaN values.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
#[derive(Debug, serde::Serialize, Deserialize, PartialEq, Clone, Copy)]
pub enum NanValueTreatment {
    /// No nan values encountered during training.
    #[serde(rename = "AsIs")]
    Unspecified,
    /// NaN values are treated as true.
    #[serde(rename = "AsTrue")]
    Left,
    /// NaN values are treated as false.
    #[serde(rename = "AsFalse")]
    Right,
}

/// An oblivious tree.
///
/// Oblivious trees are a type of decision tree that are used in CatBoost.
/// They have the property that all nodes on the same level in the tree are identical.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ObliviousTree {
    /// The leaf values of the tree.
    leaf_values: Vec<f32>,
    /// The splits of the tree.
    splits: Vec<Split>,
}

/// A split in the tree.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case", tag = "")]
pub struct Split {
    #[allow(unused)]
    /// The type of split. Currently only float features are supported
    split_type: SplitType,
    /// The index of the feature in the flat feature vector.
    #[allow(unused)]
    float_feature_index: usize,
    /// The index of the split in the tree.
    /// Technically, this is the only field in [`Split`] used for inference
    split_index: usize,
    /// The border of the feature.
    #[allow(unused)]
    border: f32,
}

/// The type of split.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
#[derive(Debug, Deserialize)]
pub enum SplitType {
    /// A split on a float feature. (Currently only float features are supported)
    #[serde(rename = "FloatFeature")]
    FloatFeature,
}

/// An error that occurs when performing inference with a CatBoost model.
#[derive(Debug, Error, serde::Serialize)]
pub enum InferenceError {
    /// The number of features provided does not match the number of features expected by the model.
    #[error("Incorrect number of features provided. Expected {expected}, got {actual}.")]
    NumFeaturesMismatch {
        /// The expected number of features.
        expected: usize,
        /// The actual number of features.
        actual: usize,
    },
}

impl CatBoost {
    /// Performs inference against the provided features.
    ///
    /// Returns the the prediction as a probability between 0 and 1.
    ///
    /// ```rust
    /// use catboost::CatBoost;
    /// use std::path::Path;
    /// let model = CatBoost::load(Path::new("models/test/tiny-binary-catboost.json")).unwrap();
    /// let test_features: Vec<f32> = vec![0.1276993, 0.9918129, 0.16597846, 0.98612934];
    /// let probability = model.predict(&test_features).unwrap(); // predict returns a probability
    /// assert!(
    ///     (probability - 0.5245).abs() < 0.01, // about a 52% chance of being positive
    ///     "Probability {probability} does not match expected value."
    /// );
    /// ```
    ///
    /// It is essential that the features be provided in the same order as they were during training.
    pub fn predict(&self, features: &[f32]) -> Result<f32, InferenceError> {
        let prediction = self.predict_raw(features)?;
        // convert from bits to probability (sigmoid function)
        let probability = 1.0 / (1.0 + (-prediction).exp());
        Ok(probability)
    }

    /// Performs inference against the provided features.
    ///
    /// Returns the the prediction in logit space (i.e. the raw output of the model).
    ///
    /// ```rust
    /// use catboost::CatBoost;
    /// use std::path::Path;
    /// let model = CatBoost::load(Path::new("models/test/tiny-binary-catboost.json")).unwrap();
    /// let test_features: Vec<f32> = vec![0.1276993, 0.9918129, 0.16597846, 0.98612934];
    /// let probability = model.predict_raw(&test_features).unwrap(); // predict returns a probability in logit space (-inf..inf)
    /// assert!(
    ///     (probability - 0.09794967).abs() < 0.01, // 0.097 corresponds to about a 52% chance of being positive
    ///     "Probability {probability} does not match expected value."
    /// );
    /// ```
    ///
    /// It is essential that the features be provided in the same order as they were during training.
    pub fn predict_raw(&self, features: &[f32]) -> Result<f32, InferenceError> {
        // Sanity check that the number of features is correct
        {
            let expected_features = self.num_features();

            if features.len() != expected_features {
                return Err(InferenceError::NumFeaturesMismatch {
                    expected: expected_features,
                    actual: features.len(),
                });
            }
        }

        let go_lefts = self.features_info.float_features.iter().flat_map(|FloatFeature {
            feature_index,
            flat_feature_index,
            borders,
            has_nans,
            nan_value_treatment,
        }| {
            // flat_feature_index: feature index in pool, zero-based indexation
            // feature_index: feature index among only float features, zero-based indexation. 
            assert_eq!(*feature_index, *flat_feature_index, "This will always be true if there are only float features (i.e. no categorical features). Categorical features are not supported.");
            let feature_value = features[*flat_feature_index];
            borders.iter().map(move |border| {
                if feature_value.is_nan() {
                    if !has_nans {
                        eprintln!(
                            "Warning: Encountered NaN for feature {} which had no NaNs during training. Treating as <= border.",
                            feature_index
                        );
                        false
                    } else {
                        // Handle NaN based on training treatment
                        match nan_value_treatment {
                            NanValueTreatment::Unspecified => {
                                eprintln!(
                                    "Warning: Encountered NaN for feature {} with NanValueTreatment::AsIs. Treating as <= border.",
                                    feature_index
                                );
                                false
                            }
                            NanValueTreatment::Left => true, // NaN goes left (like > border)
                            NanValueTreatment::Right => false, // NaN goes right (like <= border)
                        }
                    }
                } else {
                    // Standard comparison for non-NaN values
                    feature_value > *border
                }
            }
        )}).collect::<Vec<bool>>();

        let logits = self
            .oblivious_trees
            .iter()
            .map(|tree| {
                assert_eq!(
                    2_usize.pow(tree.splits.len() as u32),
                    tree.leaf_values.len(),
                    "The number of leaf values must be equal to 2^number_of_splits"
                );

                let mut current_leaf_index: usize = 0;
                let depth = tree.splits.len();

                if depth == 0 {
                    // Tree might have no splits (just a constant value)
                    if !tree.leaf_values.is_empty() {
                        return tree.leaf_values[0];
                    }
                    // this should hopefully not happen
                    panic!("No leaf values!?");
                }

                for (level, Split { split_index, .. }) in tree.splits.iter().enumerate() {
                    let go_left = go_lefts[*split_index];

                    // Set the corresponding bit in the index.
                    current_leaf_index |= (go_left as usize) << level;
                }

                tree.leaf_values[current_leaf_index]
            })
            .sum::<f32>();

        // Apply scale and bias
        let scale = self.scale_and_bias.0;
        let bias = self.scale_and_bias.1.first().unwrap_or(&0.0); // hopefully only one bias term

        let prediction = logits * scale + bias;

        Ok(prediction)
    }
}

#[test]
fn test_against_tiny_model() {
    let model = CatBoost::load(Path::new("models/test/tiny-binary-catboost.json")).unwrap();
    let test_features: Vec<f32> = vec![0.1276993, 0.9918129, 0.16597846, 0.98612934];
    let probability = model.predict(&test_features).unwrap();

    assert!(
        (probability - 0.5245).abs() < 0.01,
        "Probability does not match expected value."
    );
}

#[test]
fn test_against_big_model() {
    let model = CatBoost::load(Path::new("models/test/big-binary-catboost.json")).unwrap();

    // deserialize the test features from JSON the way rustfmt wants to format it if you just put a vec in here is horrible
    let test_features: Vec<f32> = serde_json::from_str(
        "[-7.60986700e-04,-1.16379880e-02,-1.18961320e-02,2.97898050e-01,-1.04892480e-01,-1.98598710e-01,-1.47249590e-02,
        1.38537230e-01,8.87154600e-02,4.81008140e-02,2.59864870e-02,-1.16422900e-01,6.40196900e-02,9.56853400e-02,-1.17455475e-01,
        -1.70977310e-01,-1.43097770e-01,-8.89736000e-02,-1.75322740e-02,-6.27612370e-03,6.12661540e-02,2.41794800e-01,
        5.64474700e-02,1.10313500e-01,-1.16272320e-02,-3.90173750e-03,-4.98648000e-02,3.72372570e-02,9.55132500e-02,
        4.76275500e-02,9.35341400e-02,-7.05593400e-02,-5.39090520e-02,-8.08850900e-02,-5.20859100e-03,-1.27458550e-02,
        -1.56865450e-01,-2.96650380e-02,9.99769850e-03,-6.87093100e-02,3.25046220e-02,1.51788620e-01,6.56115800e-02,
        -2.34910960e-02,-9.78365400e-02,1.23909080e-02,-3.94314830e-02,-8.03257800e-02,1.14529850e-01,2.28887600e-01,
        -1.26167830e-02,3.24831100e-02,-1.31223150e-03,1.72440130e-01,-4.61217130e-02,-5.99754340e-02,-1.60393420e-01,
        1.37332560e-01,-1.32083640e-01,4.97787500e-02,-7.21512200e-02,-3.61616570e-02,-7.18070300e-02,8.66072800e-02,-1.83454280e-01,
        -2.79655900e-02,-6.01045080e-02,-1.57725930e-01,1.21671826e-01,4.59065920e-02,2.10172160e-02,-9.08666550e-02,
        -6.02335780e-02,3.82698330e-02,3.70006260e-03,-7.22372700e-02,1.00417980e-01,1.46970600e-04,
        1.44302440e-01,4.17978000e-02,1.33804590e-01,-7.68408300e-02,-3.29993960e-02,1.02224990e-01,-1.41721010e-01,
        1.25027700e-01,-1.29502190e-01,-5.90719320e-02,-7.84757500e-02,-6.27289700e-02,-2.28199210e-01,1.31739440e-01,-2.71051100e-02,
        -5.61463000e-02,1.48174600e-01,1.09539060e-01,7.42163700e-02,-1.00729900e-02,3.70221400e-02,7.27535560e-02,
        -8.97094300e-02,6.24005540e-03,1.35485190e-01,-7.96700800e-02,-1.05367130e-01,-4.21836970e-02,9.26107100e-02,
        4.85388820e-02,-6.13413560e-02,-1.53906020e-01,3.15686950e-02,-1.97217990e-02,-5.83019200e-02,-3.23515800e-02,
        3.77396700e-02,6.65912900e-02,-9.17817700e-02,-3.21443450e-02,-8.52423800e-02,5.77953460e-02,-5.77492940e-02,2.00211370e-02,
        5.01507040e-02,1.35945710e-01,-1.11538110e-01,-5.57690560e-02,-5.82558660e-02,1.98576520e-01,8.29858260e-02,-2.80917600e-01,
        1.01130344e-01,-1.45340340e-01,4.36113100e-02,-3.87125200e-04,5.07954320e-02,1.22406400e-01,9.71698700e-02,
        7.49267200e-02,-1.03064530e-01,-1.75918900e-01,1.06288180e-01,-2.05576430e-01,1.21945880e-01,-3.35259070e-02,
        -4.77099460e-02,4.69270570e-02,-1.01775070e-01,8.87078000e-03,1.51603420e-01,1.30879980e-01,-1.06380284e-01,
        1.34356920e-02,-2.48450920e-02,1.33781270e-02,-2.55128460e-02,-2.23467670e-02,-1.91116090e-03,1.12735465e-01,-6.37821200e-02,
        5.47559100e-02,1.28301070e-01,-7.57556560e-02,-1.68233970e-03,8.44134500e-02,-8.63936840e-02,1.58879640e-01,
        3.69855670e-04,2.59042890e-02,-7.91520000e-03,6.05584700e-02,-1.23074160e-02,1.17248570e-01,4.87691420e-02,
        -5.97755870e-02,1.03893470e-01,-9.91846400e-03,-3.04404180e-02,1.81353050e-01,1.82337410e-03,-1.62103290e-02,
        3.71870470e-02,-6.09729400e-02,-6.26768700e-02,-1.42024580e-01,-1.39169350e-01,1.43498240e-01,-3.35811700e-01,
        -9.47751550e-02,2.49141700e-02,1.44258100e-02,-2.00787020e-02,1.68550580e-01,-5.71333500e-04,3.26739440e-02,
        1.65511130e-01,3.88679470e-02,-8.53114600e-03,6.17558250e-02,4.11244970e-02,2.50339060e-01
    ]",
    )
    .unwrap();

    let probability = model.predict(&test_features).unwrap();

    assert!(
        (probability - 0.74518714).abs() < 0.01,
        "Probability does not match expected value."
    );
}

#[cfg(target_arch = "wasm32")]
impl From<InferenceError> for JsValue {
    fn from(err: InferenceError) -> JsValue {
        // Convert your error to a string representation
        JsValue::from_str(&format!("{:?}", err))
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl CatBoost {
    /// Load a CatBoost model from a JSON string.
    #[wasm_bindgen(constructor)]
    pub fn load_from_string(model_str: &str) -> Result<CatBoost, String> {
        let model: CatBoost = CatBoost::try_from_json(model_str).map_err(|e| e.to_string())?;
        Ok(model)
    }

    /// Perform inference against the provided features.
    /// The number of features must match the number of features expected by the model.
    #[cfg(target_arch = "wasm32")]
    #[wasm_bindgen]
    pub fn infer(&self, features: &[f32]) -> Result<f32, InferenceError> {
        self.predict(features)
    }
}
