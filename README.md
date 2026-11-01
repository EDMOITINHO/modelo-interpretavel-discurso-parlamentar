# Análise do discurso parlamentar brasileiro com métodos de PLN

Repositório associado à dissertação de mestrado de **Edpaula Moitinho dos Anjos**, desenvolvida no Programa de Pós-Graduação em Modelagem e Simulação de Biossistemas da Universidade do Estado da Bahia (UNEB), sob orientação do Prof. Dr. Marcos Batista Figueredo.

## Objetivo

Disponibilizar os materiais computacionais que sustentam a dissertação e permitir a verificação independente dos procedimentos e resultados. O repositório reúne os scripts de análise, os dicionários lexicais, o mapeamento dos partidos em blocos ideológicos, as planilhas de validação manual, parâmetros de execução, resultados agregados e figuras finais.

## Fonte dos dados

Os discursos parlamentares foram obtidos por meio da **API de Dados Abertos da Câmara dos Deputados**:

https://dadosabertos.camara.leg.br/api/v2

Período analisado: **2003–2023**  
Legislaturas: **52ª a 57ª**  
A **57ª legislatura é parcial**, limitada aos dados disponíveis até 2023.

O script de coleta é `scripts/coletaV5.py`.

## Construção do corpus

- Registros iniciais: **449.345**
- Registros vazios removidos: **23**
- Discursos com menos de 50 palavras removidos: **9.694**
- Registro removido por possuir menos de 20 tokens após a limpeza: **1**
- Corpus final: **439.627 discursos**

A distribuição do corpus final por legislatura está em `data/corpus_manifest.csv`.

O corpus bruto completo não é versionado no GitHub. Os dados originais permanecem acessíveis na API oficial da Câmara dos Deputados e o processo de coleta e transformação está documentado pelos scripts do repositório.

## Pipeline metodológico

**API da Câmara → coleta → auditoria → filtragem → pré-processamento → quatro camadas analíticas → validação e robustez → resultados agregados → figuras**

### Coleta, auditoria e pré-processamento

- `coletaV5.py`: coleta e consolidação dos discursos.
- `00_check_estrutura_corpus.py`: auditoria da estrutura e da qualidade do corpus.
- `01_preprocess_corpus.py`: filtragem, normalização e preparação dos dados.

### Representações semânticas

- `02_train_word2vec.py`: treinamento de modelos Word2Vec por legislatura.
- `03_align_procrustes.py`: alinhamento dos espaços semânticos por Orthogonal Procrustes.
- `04_semantic_shift.py`: cálculo da mudança semântica.
- `05_interpret_semantic_shift.py`: organização dos resultados de mudança semântica.
- `06_refine_interpretable_tables.py`: refinamento das tabelas interpretáveis.
- `07_discourse_construction.py`: síntese dos eixos institucionais e trajetórias de termos.

### Camada 1 — Agendas nacionais

- `08_national_policy_themes.py`: identificação de agendas nacionais por dicionários lexicais.
- `14_dictionary_robustness_check.py`: comparação das regras R1, R2 e R3.

Nas análises longitudinais principais, a dissertação utiliza a **regra R2**, que exige pelo menos dois termos distintos da categoria ou três ocorrências totais no discurso.

### Camada 2 — Mediação setorizada

- `09_sectorized_agendas.py`: identificação da presença simultânea de agendas nacionais e referências setorizadas.

### Camada 3 — Instabilidade e divergência semântica

- `04_semantic_shift.py`: instabilidade semântica entre legislaturas adjacentes.
- `12_bootstrap_drift_ci.py`: intervalos de confiança por bootstrap.
- `13_ideological_divergence.py`: divergência semântica entre os blocos Esquerda e Direita.
- `16_word2vec_robustness_battery.py`: sementes alternativas, limiares de frequência, corpus balanceado e modelo nulo.

### Camada 4 — Grupos sociais e auditoria adversarial

- `10_adversarial_group_discourse.py`: triagem de casos candidatos pela proximidade entre referências a grupos sociais e léxicos adversariais.
- `15_adversarial_validation_sample.py`: geração da amostra estratificada utilizada na validação manual.
- `validation/`: codebook, amostra e rótulos da validação humana.

A quarta camada é uma **auditoria de casos candidatos**, e não um classificador definitivo de ataques.

## Validação e robustez

A dissertação incorpora:

- validação manual de **600 casos**, com **150 casos por grupo-alvo**;
- rótulos principais: `attack`, `defense`, `neutral` e `ambiguous`;
- bootstrap com **1.000 reamostragens**;
- sementes alternativas **7** e **123**;
- `min_count` alternativo de **50** e **100**;
- corpus balanceado por número de tokens;
- modelo nulo com divisão intra-legislatura.

## Parâmetros principais do Word2Vec

- dimensão dos vetores: **300**;
- janela de contexto: **8**;
- `min_count`: **20**;
- épocas: **10**;
- negative sampling: **10**;
- subsampling: **1e-4**;
- semente principal: **42**.

Os parâmetros consolidados estão em `config/analysis_parameters.csv`.

## Estrutura do repositório

- `data/`: manifesto e dicionário de dados do corpus.
- `scripts/`: código do pipeline metodológico.
- `dictionaries/`: dicionários lexicais das camadas de agenda, setorização e auditoria adversarial.
- `config/`: parâmetros de análise e mapeamento dos partidos em blocos ideológicos.
- `validation/`: materiais da validação manual.
- `results/tables/`: resultados agregados selecionados.
- `figures/`: figuras finais associadas aos principais resultados.
- `environment.yml`: ambiente computacional usado para reprodução das análises.
- `CITATION.cff`: informações de citação do repositório.
- `PIPELINE_METODOLOGICO.md`: ordem lógica das etapas e relação entre scripts, entradas e saídas.

## Resultados e figuras

Os resultados agregados utilizados na dissertação estão em `results/tables/`. As figuras finais estão em `figures/`, acompanhadas de um índice que relaciona cada gráfico ao respectivo resultado tabular.

## Reprodutibilidade

O ambiente computacional está fixado em `environment.yml`, com Python 3.11 e as principais bibliotecas utilizadas na dissertação. O arquivo `PIPELINE_METODOLOGICO.md` apresenta a sequência completa de execução.

## Autoria

**Edpaula Moitinho dos Anjos**  
Orientador: **Prof. Dr. Marcos Batista Figueredo**  
Coorientador: **Prof. Dr. Vagner de Souza Fonseca**

Universidade do Estado da Bahia — UNEB
