# Explicaciones contrafactuales multi-objetivo para clasificadores binarios de imágenes: un marco de generación y evaluación

Implementación experimental desarrollada como parte de la Tesis presentada a la Facultad de Ingeniería de la Universidad de la República por Ariel Malowany, en cumplimiento parcial de los requerimientos para la obtención del título de Magister en Ciencia de Datos y Aprendizaje Automático. El proyecto replica y extiende la metodología de generación de explicaciones contrafactuales multi-objetivo propuesta por Del Ser et al. (2024), aplicada a la clasificación de género sobre imágenes de rostros humanos.

## Contexto académico

El objetivo central es evaluar si las propiedades de las explicaciones contrafactuales (validez, plausibilidad, proximidad, minimalidad, diversidad, accionabilidad) son verificables de manera objetiva y cuantificable. Para ello se reemplaza el clasificador original de Del Ser et al. por **MiVOLO** (`mivolo_d1`), un modelo de estado del arte, y el conjunto de datos `CelebA` por `CelebA-HQ` en resolución 384×384. Las explicaciones se generan condicionadas a 13 atributos faciales interpretables mediante la red **Att-GAN**, y la búsqueda se conduce con el algoritmo evolutivo multi-objetivo **NSGA-II**.

**Referencia principal:** Del Ser et al., *On generating trustworthy counterfactual explanations*, Information Sciences, 2024.

## Estructura del repositorio

```
.
├── generate_gender_cfs.py          # Definición del problema de optimización (jMetalPy FloatProblem)
├── batch_generate_cfs.py           # Generación en lote de contrafactuales (CLI)
├── batch_evaluate_cfs.py           # Cálculo en lote de métricas de evaluación (CLI)
├── create_evaluation_metrics.py    # Script individual de métricas (por instancia)
├── cf_utils.py                     # Funciones auxiliares: normalización, visualización, I/O
├── data.py                         # Datasets personalizados (CelebA-HQ, Custom)
├── my_attgan.py                    # Wrapper de Att-GAN (encoder + decoder)
├── my_attgan_blocks.py             # Bloques arquitectónicos de Att-GAN
├── my_mivolo_inference.py          # Wrapper de inferencia de MiVOLO
├── attgan.py                       # Arquitectura completa de Att-GAN (con discriminador)
├── nn.py                           # Bloques de red neuronal genéricos
├── switchable_norm.py              # Normalización intercambiable (Att-GAN)
├── train_celeba_hq_classifier.py   # Entrenamiento del clasificador baseline de Del Ser et al.
├── update_minimality_metric.py     # Recálculo de la métrica de minimalidad
├── run_discriminative_power_all_samples.py  # Evaluación de capacidad discriminativa (1-NN)
├── mivolo/                         # Código fuente de MiVOLO (submódulo)
├── ATT_GAN_Discovery.ipynb         # Análisis exploratorio de Att-GAN
├── MiVOLO_Discovery.ipynb          # Análisis exploratorio de MiVOLO y evaluación sobre datasets
├── NSGAII_Algorithm.ipynb          # Exploración del algoritmo NSGA-II
├── Generate_Gender_CFs.ipynb       # Generación de contrafactuales (notebook interactivo)
├── CF_Evaluation_Metrics.ipynb     # Cálculo de métricas de evaluación
├── Experiment_Results.ipynb        # Análisis de resultados finales
├── Visualizaciones.ipynb           # Visualización de contrafactuales y frentes de Pareto
└── requirements.txt
```

### Carpetas no versionadas (ver `.gitignore`)

| Carpeta / archivo | Contenido |
|---|---|
| `384_shortcut1_inject1_none_hq/` | Pesos de Att-GAN y archivo de configuración `setting.txt` |
| `mivolo_models/` | Pesos de MiVOLO (`model_imdb_cross_person_4.22_99.46.pth.tar`) y detector YOLOv8 (`yolov8x_person_face.pt`) |
| `celeba_hq_dataset/` | Imágenes de CelebA-HQ y archivo de anotaciones de atributos |
| `Counterfactuals/` | Resultados generados (archivos `.pkl` por instancia factual) |
| `graficos/` | Gráficos exportados de los notebooks |
| `classifier_model.h5` | Pesos del clasificador baseline de Del Ser et al. |

---

## Requisitos

- Python 3.10+
- CPU (los experimentos se ejecutaron en CPU; GPU no es obligatoria pero acelera la inferencia de MiVOLO)
- ~20 GB de espacio libre para pesos y dataset
- ~30 min por instancia factual en CPU (generación + evaluación)

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd COUNT-GEN-REPLICABILITY-REPO
```

### 2. Crear el entorno virtual e instalar dependencias

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Descargar los pesos de Att-GAN

Los pesos corresponden a la variante `384_shortcut1_inject1_none_hq` entrenada sobre CelebA-HQ, disponibles en el repositorio público [AttGAN-PyTorch](https://github.com/elvisyjlin/AttGAN-PyTorch).

Ubicar los archivos descargados en:

```
384_shortcut1_inject1_none_hq/
├── setting.txt
└── weights.149.pth
```

### 4. Descargar los pesos de MiVOLO

Los pesos se obtienen desde el repositorio oficial [MiVOLO](https://github.com/WildChlamydia/MiVOLO).

- Modelo: `model_imdb_cross_person_4.22_99.46.pth.tar`
- Detector: `yolov8x_person_face.pt`

Ubicar ambos archivos en:

```
mivolo_models/
├── model_imdb_cross_person_4.22_99.46.pth.tar
└── yolov8x_person_face.pt
```

### 5. Descargar el dataset CelebA-HQ

El dataset se puede obtener desde [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ). Se requieren las imágenes en resolución 1024×1024 (el pipeline las redimensiona internamente a 384×384) y el archivo de anotaciones de atributos.

Ubicar los archivos en:

```
celeba_hq_dataset/
├── CelebA-HQ-img/          # Imágenes .jpg
└── CelebAMask-HQ-attribute-anno.txt
```

---

## Uso

### Generar contrafactuales

El script principal acepta tres modos:

```bash
# Generar para todas las instancias existentes en Counterfactuals/
python batch_generate_cfs.py

# Generar para N instancias aleatorias
python batch_generate_cfs.py --batch-size 5

# Generar para instancias específicas (por índice de imagen)
python batch_generate_cfs.py --samples 1088 9731
```

Los resultados se guardan en `Counterfactuals/` como archivos `.pkl` con el frente de Pareto, la imagen factual, los atributos y metadatos de ejecución.

### Evaluar métricas

```bash
python batch_evaluate_cfs.py
```

Calcula el conjunto completo de métricas de evaluación (validez, disimilaridad, plausibilidad, accionabilidad, implausibilidad, capacidad contrastiva, inestabilidad, composición, hipervolumen) sobre todos los archivos `.pkl` en `Counterfactuals/`.

---

## Diseño experimental

### Componentes del sistema

| Componente | Detalle |
|---|---|
| **Generador** | Att-GAN `384_shortcut1_inject1_none_hq`, entrenado sobre CelebA-HQ a 384×384 |
| **Clasificador auditado** | MiVOLO `mivolo_d1` con pesos `model_imdb_cross_person_4.22_99.46` |
| **Algoritmo de búsqueda** | NSGA-II (jMetalPy): población 100, SBX crossover (p=0.9), mutación polinomial (p=1/13) |
| **Atributos de decisión** | 13 atributos faciales de CelebA-HQ, rango continuo [−1, 1] |
| **Criterio de parada** | Máximo 1100 evaluaciones o convergencia del hipervolumen |

### Funciones objetivo minimizadas

| Función | Propiedad aproximada |
|---|---|
| `f_att`: norma ℓ₂ entre atributos factual y contrafactual | Proximidad y minimalidad |
| `f_adv`: diferencia absoluta entre predicción deseada y predicción del modelo | Validez |
| `f_gan`: diferencia LPIPS entre imagen factual y contrafactual | Plausibilidad perceptual |

### Modificaciones respecto a Del Ser et al. (2024)

1. **Clasificador:** reemplazo del clasificador CNN entrenado sobre 900 imágenes (precisión ~azar en test) por MiVOLO (precisión >98% en CelebA-HQ a 384×384).
2. **Dataset:** cambio de CelebA a 128×128 por CelebA-HQ a 384×384 para garantizar compatibilidad con MiVOLO.
3. **Plausibilidad:** sustitución de la salida del discriminador GAN por la métrica LPIPS (AlexNet), dado que los pesos del discriminador no están disponibles en el repositorio público de Att-GAN.

---

## Preguntas de investigación

1. ¿Son los resultados de Del Ser et al. robustos frente a la sustitución del clasificador por un modelo de estado del arte?
2. ¿Son las propiedades de las explicaciones contrafactuales verificables de manera objetiva mediante el conjunto de métricas de Guidotti (2024), adaptado al dominio de imágenes?
3. ¿Permiten los cambios sugeridos en los atributos de generación aproximar la relevancia de cada atributo en la decisión del modelo auditado?

---

## Citas

```bibtex

@article{DELSER2024119898,
title = {On generating trustworthy counterfactual explanations},
journal = {Information Sciences},
volume = {655},
pages = {119898},
year = {2024},
issn = {0020-0255},
doi = {https://doi.org/10.1016/j.ins.2023.119898},
url = {https://www.sciencedirect.com/science/article/pii/S0020025523014834},
author = {Javier {Del Ser} and Alejandro Barredo-Arrieta and Natalia Díaz-Rodríguez and Francisco Herrera and Anna Saranti and Andreas Holzinger},
keywords = {Explainable artificial intelligence, Deep learning, Counterfactual explanations, Generative adversarial networks, Multi-objective optimization},
abstract = {Deep learning models like chatGPT exemplify AI success but necessitate a deeper understanding of trust in critical sectors. Trust can be achieved using counterfactual explanations, which is how humans become familiar with unknown processes; by understanding the hypothetical input circumstances under which the output changes. We argue that the generation of counterfactual explanations requires several aspects of the generated counterfactual instances, not just their counterfactual ability. We present a framework for generating counterfactual explanations that formulate its goal as a multiobjective optimization problem balancing three objectives: plausibility; the intensity of changes; and adversarial power. We use a generative adversarial network to model the distribution of the input, along with a multiobjective counterfactual discovery solver balancing these objectives. We demonstrate the usefulness of six classification tasks with image and 3D data confirming with evidence the existence of a trade-off between the objectives, the consistency of the produced counterfactual explanations with human knowledge, and the capability of the framework to unveil the existence of concept-based biases and misrepresented attributes in the input domain of the audited model. Our pioneering effort shall inspire further work on the generation of plausible counterfactual explanations in real-world scenarios where attribute-/concept-based annotations are available for the domain under analysis.}
}

@ARTICLE{celeba_hq,
  title         = "Progressive growing of {GANs} for improved quality,
                   stability, and variation",
  author        = "Karras, Tero and Aila, Timo and Laine, Samuli and Lehtinen,
                   Jaakko",
  abstract      = "We describe a new training methodology for generative
                   adversarial networks. The key idea is to grow both the
                   generator and discriminator progressively: starting from a
                   low resolution, we add new layers that model increasingly
                   fine details as training progresses. This both speeds the
                   training up and greatly stabilizes it, allowing us to
                   produce images of unprecedented quality, e.g., CelebA images
                   at 1024^2. We also propose a simple way to increase the
                   variation in generated images, and achieve a record
                   inception score of 8.80 in unsupervised CIFAR10.
                   Additionally, we describe several implementation details
                   that are important for discouraging unhealthy competition
                   between the generator and discriminator. Finally, we suggest
                   a new metric for evaluating GAN results, both in terms of
                   image quality and variation. As an additional contribution,
                   we construct a higher-quality version of the CelebA dataset.",
  month         =  oct,
  year          =  2017,
  copyright     = "http://arxiv.org/licenses/nonexclusive-distrib/1.0/",
  archivePrefix = "arXiv",
  primaryClass  = "cs.NE",
  eprint        = "1710.10196",
  journal={arXiv preprint arXiv:1710.10196}
}

@article{lin2021fpage,
      title={FP-Age: Leveraging Face Parsing Attention for Facial Age Estimation in the Wild}, 
      author={Yiming Lin and Jie Shen and Yujiang Wang and Maja Pantic},
      year={2021},
      eprint={2106.11145},
      journal={arXiv},
      primaryClass={cs.CV}
}

@article{mivolo2024,
   Author = {Maksim Kuprashevich and Grigorii Alekseenko and Irina Tolstykh},
   Title = {Beyond Specialization: Assessing the Capabilities of MLLMs in Age and Gender Estimation},
   Year = {2024},
   Eprint = {arXiv:2403.02302},
}

```
