# Pipeline metodológico da dissertação

Este documento apresenta a sequência computacional da dissertação **Análise do discurso parlamentar brasileiro com métodos de PLN**.

## Fluxo geral

**API da Câmara → coleta → auditoria → pré-processamento → Camada 1 → Camada 2 → Camada 3 → Camada 4 → validação e robustez → resultados → figuras**

## Ordem de execução

| Ordem | Script | Função principal |
|---|---|---|
| 1 | `coletaV5.py` | Coleta os discursos pela API oficial da Câmara e consolida o corpus bruto. |
| 2 | `00_check_estrutura_corpus.py` | Audita estrutura, distribuição, duplicatas, textos curtos e cobertura de metadados. |
| 3 | `01_preprocess_corpus.py` | Aplica os filtros e o pré-processamento definidos na dissertação e organiza os dados por legislatura e bloco ideológico. |
| 4 | `02_train_word2vec.py` | Treina modelos Word2Vec independentes por legislatura. |
| 5 | `03_align_procrustes.py` | Alinha os espaços semânticos por Orthogonal Procrustes. |
| 6 | `04_semantic_shift.py` | Calcula a distância cosseno e a instabilidade semântica entre legislaturas. |
| 7 | `05_interpret_semantic_shift.py` | Organiza termos e transições para interpretação. |
| 8 | `06_refine_interpretable_tables.py` | Refina tabelas de mudança semântica. |
| 9 | `07_discourse_construction.py` | Resume eixos institucionais e trajetórias semânticas. |
| 10 | `08_national_policy_themes.py` | Aplica os dicionários de agendas nacionais. |
| 11 | `09_sectorized_agendas.py` | Identifica a mediação setorizada das agendas. |
| 12 | `10_adversarial_group_discourse.py` | Seleciona casos candidatos na camada de grupos sociais. |
| 13 | `11_generate_empirical_figures.py` | Produz visualizações empíricas complementares. |
| 14 | `12_bootstrap_drift_ci.py` | Calcula intervalos de confiança bootstrap para a deriva semântica. |
| 15 | `13_ideological_divergence.py` | Calcula a divergência semântica entre Esquerda e Direita por legislatura. |
| 16 | `14_dictionary_robustness_check.py` | Verifica a sensibilidade das agendas às regras R1, R2 e R3. |
| 17 | `15_adversarial_validation_sample.py` | Gera a amostra estratificada de 600 casos para validação manual. |
| 18 | `16_word2vec_robustness_battery.py` | Executa as verificações de robustez dos modelos semânticos. |

## Camada 1 — Agendas nacionais

Os discursos são avaliados por dicionários lexicais em 11 categorias temáticas. Três regras de sinalização são calculadas:

- **R1:** pelo menos um termo da categoria;
- **R2:** pelo menos dois termos distintos ou três ocorrências totais;
- **R3:** presença temática sob R1 e densidade igual ou superior à mediana específica da categoria.

A **R2** é utilizada nas análises longitudinais principais.

## Camada 2 — Mediação setorizada

As agendas nacionais são relacionadas a vocabulários setorizados referentes a territórios e regiões, grupos sociais, categorias profissionais, setores econômicos, segurança, religião e outros segmentos definidos nos dicionários do repositório.

## Camada 3 — Instabilidade e divergência semântica

Modelos Word2Vec independentes são treinados por legislatura. Os espaços são alinhados por Orthogonal Procrustes e comparados por distância cosseno. A mesma lógica de representação distribucional é utilizada para comparar os blocos Esquerda e Direita dentro de cada legislatura.

Parâmetros principais: dimensão 300, janela 8, `min_count=20`, 10 épocas, negative sampling 10, subsampling 1e-4 e semente 42.

## Camada 4 — Grupos sociais e auditoria adversarial

A camada examina a proximidade entre referências a quatro grupos-alvo e léxicos associados à deslegitimação, ameaça/criminalização, restrição de direitos e pânico moral. A janela utilizada é de 12 tokens.

Os casos selecionados são candidatos à inspeção humana. A classificação manual utiliza os rótulos `attack`, `defense`, `neutral` e `ambiguous`.

## Validação e robustez

- 600 casos validados manualmente, 150 por grupo;
- bootstrap com 1.000 reamostragens;
- sementes 7 e 123 como alternativas à semente 42;
- `min_count` 50 e 100 como limiares alternativos;
- balanceamento por número de tokens;
- modelo nulo por divisão intra-legislatura.

## Produtos do pipeline

- `dictionaries/`: léxicos usados nas camadas analíticas;
- `config/`: parâmetros e classificação partidária;
- `validation/`: amostra, codebook e rótulos manuais;
- `results/tables/`: resultados agregados;
- `figures/`: gráficos finais.
