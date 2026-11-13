# Third-party acknowledgements

This benchmark builds on [ForestDiffusion](https://github.com/SamsungSAILMontreal/ForestDiffusion), accompanying *Generating and Imputing Tabular Data via Diffusion and Flow-based XGBoost Models* by Alexia Jolicoeur-Martineau, Kilian Fatras, and Tal Kachman. The bundled baseline implementations and parts of the dataset loading, evaluation, and analysis code were adapted from that repository.

## Bundled implementations

- **STaSy** ([`STaSy/`](STaSy/)): sourced through ForestDiffusion. Several files include Google Research copyright notices and Apache-2.0 license headers.
- **CTABGANPlus** ([`CTABGANPlus/`](CTABGANPlus/)): sourced through ForestDiffusion and includes local adaptations. No standalone license file is included in this directory.
- **TabDDPM** ([`TabDDPM/`](TabDDPM/)): sourced through ForestDiffusion. See its [README](TabDDPM/README.md) and [MIT license](TabDDPM/LICENSE.md), copyright Akim Kotelnikov (2022).

## Licensing and citation

The root [MIT License](LICENSE) covers this project's original contributions. Third-party code remains subject to its own licenses and copyright notices; the root license does not establish distribution rights for code whose terms are unspecified.

When using this benchmark in research, cite the accompanying FM-Tabular paper, ForestDiffusion, and the baseline methods used in your experiments. See the [ForestDiffusion references](https://github.com/SamsungSAILMontreal/ForestDiffusion#references) and bundled documentation for citation details.
