# Vision-Based Object Classifier (Fruits360)
# AI/ML Developer Assignment - Option E
# Author: [Your Name]
# Date: June 2025

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 1. Dataset paths (already extracted)
# Since dataset is already extracted, we'll work directly with the folders

# Update these paths based on your current structure
# Since you've already extracted the dataset, point directly to the folders
train_dir = "./fruits-360/Training"
test_dir = "./fruits-360/Test"

# Verify the paths exist
if os.path.exists(train_dir):
    print(f"✅ Training directory found: {train_dir}")
else:
    print(f"❌ Training directory not found: {train_dir}")

if os.path.exists(test_dir):
    print(f"✅ Test directory found: {test_dir}")
else:
    print(f"❌ Test directory not found: {test_dir}")

# 2. Data Exploration
def explore_dataset(train_dir):
    """Explore the dataset structure"""
    if os.path.exists(train_dir):
        classes = os.listdir(train_dir)
        print(f"Number of items in training directory: {len(classes)}") # Changed print statement
        print(f"First 10 items: {classes[:10]}") # Changed print statement

        # Count samples per class
        class_counts = {}
        valid_classes = [] # Keep track of actual directories
        print("\nChecking content in first 10 items:") # Added print for clarity
        for item_name in classes[:10]:  # Limit to first 10 for display
            item_path = os.path.join(train_dir, item_name)
            if os.path.isdir(item_path):
                valid_classes.append(item_name) # Add to valid classes if it's a directory
                count = len(os.listdir(item_path))
                class_counts[item_name] = count
                print(f"Found directory '{item_name}' with {count} images") # Added print for clarity
            else:
                print(f"Skipping non-directory item: {item_name}") # Added print for clarity


        print("\nSample counts for first 10 *directories*:") # Changed print statement
        for class_name, count in class_counts.items():
            print(f"{class_name}: {count} images")
        return valid_classes # Return only the names of valid directories
    
    return []

# Explore dataset
# The explore_dataset function now returns the list of identified class directories
identified_classes = explore_dataset(train_dir)

# 3. Preprocessing with Data Augmentation
IMG_SIZE = (100, 100)
BATCH_SIZE = 32

# Training data with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    validation_split=0.2  # Use 20% for validation
)

# Test data (no augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
if os.path.exists(train_dir) and os.path.exists(test_dir): # Ensure both directories exist before creating generators
    try: # Added try-except block to catch generator errors early
        train_data = train_datagen.flow_from_directory(
            train_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='training'
        )

        val_data = train_datagen.flow_from_directory(
            train_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='validation'
        )

        test_data = test_datagen.flow_from_directory(
            test_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )

        NUM_CLASSES = train_data.num_classes
        print(f"\nNumber of classes detected by generators: {NUM_CLASSES}") # Clarified output
        print(f"Training samples: {train_data.samples}")
        print(f"Validation samples: {val_data.samples}")
        print(f"Test samples: {test_data.samples}")
        print(f"Class indices (mapping): {train_data.class_indices}") # Print class indices
        if NUM_CLASSES == 0:
             print("🚨 Error: No classes (subdirectories with images) found by ImageDataGenerator in the training directory.")
        if NUM_CLASSES == 1:
            print("⚠️ Warning: Only one class detected. Make sure your training directory contains multiple subdirectories named after classes.")


    except Exception as e:
        print(f"❌ Error creating data generators: {e}")
        # Exit or handle the error appropriately if generators can't be created
        train_data = None
        val_data = None
        test_data = None
        NUM_CLASSES = 0 # Set NUM_CLASSES to 0 if creation failed


# 4. Model Building using MobileNetV2
# Only proceed if NUM_CLASSES is greater than 1 (binary or multi-class classification)
if 'NUM_CLASSES' in locals() and NUM_CLASSES > 1:
    def create_model(num_classes):
        """Create MobileNetV2-based model"""
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(100, 100, 3)
        )
        base_model.trainable = False  # Freeze base model initially

        # Add custom classifier
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(num_classes, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=predictions)
        return model, base_model

    model, base_model = create_model(NUM_CLASSES)

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Model created successfully!")
    print(f"Total parameters: {model.count_params():,}")
else:
    print("\nSkipping model creation and training: Not enough classes detected (need > 1).")
    model = None # Ensure model is None if not created


# 5. Training with Callbacks
# Only train if the model and data generators were successfully created
if 'model' in locals() and model is not None and train_data is not None and val_data is not None:
    def train_model(model, train_data, val_data, epochs=15):
        """Train the model with callbacks"""
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]

        print("Starting training...")
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        return history

    try: # Added try-except for training
        history = train_model(model, train_data, val_data)
        print("Training finished.")
    except UnidentifiedImageError as e:
         print(f"❌ UnidentifiedImageError during training: {e}")
         print("This likely means there is a non-image file in your training or validation directories.")
         print("Please check the subdirectories within './fruits-360/Training' for any files that are not standard image formats (like .jpg, .png).")
         history = None # Ensure history is None if training failed
    except Exception as e:
        print(f"❌ An unexpected error occurred during training: {e}")
        history = None # Ensure history is None if training failed

else:
    print("\nSkipping training.")
    history = None # Ensure history is None if training was skipped

# 6. Training Visualization
if 'history' in locals() and history is not None: # Only plot if history exists
    def plot_training_history(history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)

        # Loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    plot_training_history(history)
else:
    print("\nSkipping training visualization.")


# 7. Model Evaluation
# Only evaluate if model and test_data exist and NUM_CLASSES > 1
if 'model' in locals() and model is not None and 'test_data' in locals() and test_data is not None and NUM_CLASSES > 1:
    def evaluate_model(model, test_data):
        """Comprehensive model evaluation"""
        print("Evaluating model...")

        # Basic evaluation
        try: # Added try-except for evaluation
            loss, accuracy = model.evaluate(test_data, verbose=0)
            print(f"Test Loss: {loss:.4f}")
            print(f"Test Accuracy: {accuracy:.4f}")

            # Predictions
            print("Making predictions...")
            pred_probs = model.predict(test_data, verbose=0)
            y_pred = np.argmax(pred_probs, axis=1)
            y_true = test_data.classes

            return y_true, y_pred, pred_probs
        except Exception as e:
            print(f"❌ Error during model evaluation: {e}")
            return None, None, None # Return None if evaluation fails


    y_true, y_pred, pred_probs = evaluate_model(model, test_data)
else:
    print("\nSkipping model evaluation.")
    y_true, y_pred, pred_probs = None, None, None # Ensure these are None


# 8. Confusion Matrix and Classification Report
# Only plot/report if evaluation was successful and NUM_CLASSES > 1
if 'y_true' in locals() and y_true is not None and NUM_CLASSES > 1:
    def plot_confusion_matrix(y_true, y_pred, class_names, max_classes=20):
        """Plot confusion matrix (limited to first max_classes for readability)"""

        # Limit to first max_classes for visualization
        if len(class_names) > max_classes:
            print(f"Showing confusion matrix for first {max_classes} classes only")

            # Filter predictions and true labels for first max_classes
            # Need to use the original class indices for filtering
            class_indices_list = list(test_data.class_indices.values()) # Get numerical indices
            mask = np.isin(y_true, class_indices_list[:max_classes]) & np.isin(y_pred, class_indices_list[:max_classes]) # Use np.isin for filtering
            y_true_filtered = y_true[mask]
            y_pred_filtered = y_pred[mask]
            class_names_filtered = list(test_data.class_indices.keys())[:max_classes] # Use keys for names

            # Remap filtered y_true and y_pred to 0-based indices for the smaller matrix
            unique_filtered_classes = sorted(list(set(np.concatenate((y_true_filtered, y_pred_filtered)))))
            mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_filtered_classes)}
            y_true_filtered_remapped = np.array([mapping[idx] for idx in y_true_filtered])
            y_pred_filtered_remapped = np.array([mapping[idx] for idx in y_pred_filtered])

            # Update class names list based on remapped indices
            # This part needs care to align correctly with the remapped indices
            # Let's rebuild the class names based on the unique filtered classes
            class_names_remapped = [list(test_data.class_indices.keys())[idx] for idx in unique_filtered_classes]

            cm = confusion_matrix(y_true_filtered_remapped, y_pred_filtered_remapped)

            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d',
                        xticklabels=class_names_remapped, # Use remapped names
                        yticklabels=class_names_remapped, # Use remapped names
                        cmap='Blues')
            plt.title(f"Confusion Matrix (First {len(class_names_remapped)} Classes)") # Update title
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.show()

        else: # Plot full matrix if <= max_classes
            cm = confusion_matrix(y_true, y_pred)
            class_names_full = list(test_data.class_indices.keys())

            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d',
                        xticklabels=class_names_full,
                        yticklabels=class_names_full,
                        cmap='Blues')
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.show()

        return cm


    # Check if test_data is available before accessing class_indices
    if 'test_data' in locals() and test_data is not None:
        cm = plot_confusion_matrix(y_true, y_pred, test_data.class_indices)

        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred,
                                  target_names=list(test_data.class_indices.keys()),
                                  digits=3))
    else:
        print("\nSkipping confusion matrix and classification report: Test data not available.")

else:
    print("\nSkipping confusion matrix and classification report.")


# 9. Sample Predictions Visualization
# Only visualize if model and test_data exist and NUM_CLASSES > 1
if 'model' in locals() and model is not None and 'test_data' in locals() and test_data is not None and NUM_CLASSES > 1:
    def visualize_predictions(model, test_data, num_samples=8):
        """Visualize sample predictions"""
        try: # Added try-except for visualization
            # Get a batch of test images
            test_batch = next(iter(test_data))
            images, labels = test_batch

            # Make predictions
            predictions = model.predict(images)
            predicted_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(labels, axis=1)

            # Get class names
            class_names = list(test_data.class_indices.keys())

            # Plot samples
            plt.figure(figsize=(15, 10))
            for i in range(min(num_samples, len(images))):
                plt.subplot(2, 4, i + 1)
                plt.imshow(images[i])

                true_label = class_names[true_classes[i]]
                pred_label = class_names[predicted_classes[i]]
                confidence = predictions[i][predicted_classes[i]]

                color = 'green' if true_classes[i] == predicted_classes[i] else 'red'
                plt.title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}',
                         color=color, fontsize=10)
                plt.axis('off')

            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"❌ Error during prediction visualization: {e}")


    visualize_predictions(model, test_data)
else:
    print("\nSkipping sample predictions visualization.")


# 10. Model Summary and Performance Metrics
# Only print summary if model exists
if 'model' in locals() and model is not None:
    def print_model_summary(model, history=None):
        """Print comprehensive model summary"""
        print("="*50)
        print("MODEL SUMMARY")
        print("="*50)

        print(f"Model Architecture: MobileNetV2 + Custom Classifier")
        print(f"Input Shape: {model.input_shape}")
        print(f"Output Classes: {model.output_shape[-1]}")
        print(f"Total Parameters: {model.count_params():,}")

        if history:
            # Ensure history contains required keys before accessing
            if 'accuracy' in history.history and 'val_accuracy' in history.history:
                final_train_acc = history.history['accuracy'][-1]
                final_val_acc = history.history['val_accuracy'][-1]
                best_val_acc = max(history.history['val_accuracy'])

                print(f"\nTraining Results:")
                print(f"Final Training Accuracy: {final_train_acc:.4f}")
                print(f"Final Validation Accuracy: {final_val_acc:.4f}")
                print(f"Best Validation Accuracy: {best_val_acc:.4f}")
                print(f"Total Epochs Trained: {len(history.history['accuracy'])}")
            else:
                 print("\nTraining history is incomplete or missing expected keys.")
        else:
            print("\nNo training history available.")

    print_model_summary(model, history if 'history' in locals() else None)
else:
    print("\nSkipping model summary.")


print("\n" + "="*50)
print("ANALYSIS COMPLETE!")
print("="*50)