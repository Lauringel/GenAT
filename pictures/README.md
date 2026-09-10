# Image Attribution

The images contained in this directory are cropped details of public-domain artworks from the National Gallery of Art Open Access collection. They are included solely to illustrate the creation of the synthetic dataset corresponding to **Figure 3** of the accompanying paper.

Each synthetic image was created by overlaying a manually annotated crack mask from an existing crack dataset onto a cropped painting detail (see synthetic_data.py).

## File mapping

| File | Description | Source | License |
|---|---|---|---|
| `01784.jpg` | Cropped painting detail | *Sweet Tremulous Leaves* by Arthur B. Davies, 1922–1923, courtesy National Gallery of Art, Washington, DC, URL: https://www.nga.gov/artworks/46594-sweet-tremulous-leaves | [CC0&nbsp;1.0](https://creativecommons.org/publicdomain/zero/1.0/) |
| `01848.jpg` | Cropped painting detail | *The Fire Boss* by George Luks, 1925, courtesy National Gallery of Art, Washington, DC, URL: https://www.nga.gov/artworks/42925-fire-boss | [CC0&nbsp;1.0](https://creativecommons.org/publicdomain/zero/1.0/) |
| `01883.jpg` | Cropped painting detail | *Portrait of Vincent van Gogh* (imitator of Vincent van Gogh), 1925–1928, courtesy National Gallery of Art, Washington, DC, URL: https://www.nga.gov/artworks/46628-portrait-vincent-van-gogh | [CC0&nbsp;1.0](https://creativecommons.org/publicdomain/zero/1.0/) |
| `mask_01783.png` | Ground-truth crack mask | CrackLS315 / CRKWH100 (DeepCrack) dataset | [CC&nbsp;BY&nbsp;4.0](https://creativecommons.org/licenses/by/4.0/) |
| `mask_01847.png` | Ground-truth crack mask | CrackLS315 / CRKWH100 (DeepCrack) dataset | [CC&nbsp;BY&nbsp;4.0](https://creativecommons.org/licenses/by/4.0/) |
| `mask_01882.png` | Ground-truth crack mask | CrackLS315 / CRKWH100 (DeepCrack) dataset | [CC&nbsp;BY&nbsp;4.0](https://creativecommons.org/licenses/by/4.0/) |
| `synthetic_01783.png` | Synthetic image generated from `01784.jpg` and `mask_01783.png` | This work | [CC&nbsp;BY&nbsp;4.0](https://creativecommons.org/licenses/by/4.0/) |
| `synthetic_01847.png` | Synthetic image generated from `01848.jpg` and `mask_01847.png` | This work | [CC&nbsp;BY&nbsp;4.0](https://creativecommons.org/licenses/by/4.0/) |
| `synthetic_01882.png` | Synthetic image generated from `01883.jpg` and `mask_01882.png` | This work | [CC&nbsp;BY&nbsp;4.0](https://creativecommons.org/licenses/by/4.0/) |
| `Brugghen_patch.jpg` | Cropped painting detail provided by the Doerner Institut | [Hendrick ter Brugghen, *Der Zecher*, 1627](https://www.sammlung.pinakothek.de/de/artwork/8eGVD2rxWQ), Bayerische Staatsgemäldesammlungen – Alte Pinakothek | [CC&nbsp;BY&#8209;SA&nbsp;4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| `Lorrain_patch.jpg` | Cropped painting detail provided by the Doerner Institut | [Claude Lorrain (Claude Gellée), *Hagar und Ismael in der Wüste*, 1668](https://www.sammlung.pinakothek.de/de/artwork/ApL8qq2GN2), Bayerische Staatsgemäldesammlungen – Alte Pinakothek München | [CC&nbsp;BY&#8209;SA&nbsp;4.0](https://creativecommons.org/licenses/by-sa/4.0/) |

The original National Gallery of Art images are provided under the **CC0 1.0 Universal (Public Domain Dedication)**. The cropped painting details derived from them remain in the public domain.

The included crack masks originate from the **CrackLS315** and **CRKWH100 (DeepCrack)** datasets and are redistributed with permission from the DeepCrack authors under CC BY 4.0. The synthetic images were created by combining the respective public-domain painting detail with the corresponding crack mask and are also licensed under CC BY 4.0.

`Brugghen_patch.jpg` and `Lorrain_patch.jpg` are excluded from the repository’s general CC BY 4.0 license and remain licensed under **CC BY-SA 4.0**.
