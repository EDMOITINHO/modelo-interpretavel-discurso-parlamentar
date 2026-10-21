# Validação humana

Esta pasta contém os materiais da validação manual da camada de grupos sociais.

## Amostra

A validação foi realizada sobre uma amostra estratificada de **600 casos**, com **150 casos por grupo-alvo**.

## Rótulos principais

- `attack`
- `defense`
- `neutral`
- `ambiguous`

## Interpretação

A seleção automática identifica casos em que referências a grupos sociais aparecem próximas a léxicos adversariais. Essa seleção é uma triagem de candidatos e não uma classificação definitiva de ataques. A leitura humana distingue ataque, defesa, menção neutra e contexto ambíguo.

## Conteúdo

- `discourse_labels_codebook.csv`: esquema de anotação.
- `adversarial_validation_sample.csv`: amostra selecionada.
- `adversarial_validation_sample_labeled.csv`: rótulos atribuídos aos casos auditados.
