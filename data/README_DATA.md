# Dados e construção do corpus

## Fonte

Os discursos parlamentares foram coletados pela API de Dados Abertos da Câmara dos Deputados:

https://dadosabertos.camara.leg.br/api/v2

## Recorte

- Período: 2003–2023.
- Legislaturas: 52ª a 57ª.
- A 57ª legislatura é parcial, limitada aos dados disponíveis até 2023.

## Construção do corpus

- Registros iniciais: 449.345.
- Registros vazios removidos: 23.
- Discursos com menos de 50 palavras removidos: 9.694.
- Registro com menos de 20 tokens após a limpeza removido: 1.
- Corpus final: 439.627 discursos.

## Pré-processamento

O processamento inclui conversão para minúsculas, remoção de acentos, URLs, caracteres não alfabéticos e stopwords em português, tokenização por expressão regular e remoção de tokens com menos de três caracteres. Não foram aplicados lematização nem stemming.

## Arquivos

- `corpus_manifest.csv`: distribuição do corpus final por legislatura.
- `data_dictionary.csv`: descrição dos campos utilizados no pipeline.

O corpus bruto completo não é versionado no GitHub. Os dados originais são públicos e permanecem acessíveis na API oficial da Câmara dos Deputados; os scripts `coletaV5.py` e `01_preprocess_corpus.py` documentam sua reconstrução e processamento.
