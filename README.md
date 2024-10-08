# FlashSpeech







Implementation of the [FlashSpeech](https://arxiv.org/abs/2404.14700). For all details check out our paper accepted to ACM MM 2024: FlashSpeech: Efficient Zero-Shot Speech Synthesis.



## Notice

- This project is a modified version based on Amphion's **NaturalSpeech2** due to the use of some internal Microsoft tools in the original code.
- **Environment Setup**:
  ```bash
  bash env.sh
  ```
- I have replaced Amphion's `accelerate` with `lightning` because I encountered similar issues ([related issue](https://github.com/open-mmlab/Amphion/issues/120)). Training with `lightning` is faster.

## Data Preparation

- Modify `ns2dataset.py` based on your data.
- This version has been tested on the **LibriTTS** dataset. Ensure you have the following data prepared in advance:
  - Pitch
  - Code
  - Phoneme
  - Duration

## Training

1. **Run the Training Script**:
   ```bash
   bash egs/tts/NaturalSpeech2/run_train.sh
   ```

**Important Notes**:

- **Choose Configuration**:
  - You can select either `***_s1` or `***_s2` configuration files based on the training stage.
  
- **Modify Model Codec**:
  - In `models/tts/naturalspeech2/flashspeech.py`, update the codec to your own.
  - Adjust `self.latent_norm` to normalize the codec latent to the standard deviation. *(This step is crucial for training the consistency model.)*
  
- **Stage 2 Setup**:
  - In `models/tts/naturalspeech2/flashspeech_trainer_stage2.py`, set the initial weights obtained from Stage 1 training.
  
- **Stage 3 Development**:
  - The code for Stage 3 is not yet released. However, you can refer to Stage 1's consistency training to implement it.

## TODO
Further organize the project structure and complete the remaining code.

## Acknowledgements

Special thanks to **Amphion**, as our codebase is primarily borrowed from Amphion.

 


Thank you for using FlashSpeech!


 