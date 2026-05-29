# Contribuindo com o CopIA ⚽🤖

Obrigado por querer contribuir! O CopIA é um projeto educacional open-source — toda contribuição, grande ou pequena, é bem-vinda.

## Formas de Contribuir

### 🐛 Reportar Bugs
Abra uma [issue](https://github.com/ahirtonlopes/copa-ia/issues) com:
- Descrição clara do problema
- Passos para reproduzir
- Comportamento esperado vs. observado
- Versão do Python e sistema operacional

### 💡 Sugerir Melhorias
Abra uma issue com a label `enhancement` descrevendo:
- O que você gostaria de ver
- Por que seria útil para o projeto

### 📊 Melhorar os Dados
- Adicionar novas features relevantes
- Corrigir dados históricos incorretos
- Integrar novas fontes (FBref, StatsBomb, etc.)

### 🧠 Melhorar os Modelos
- Implementar novos modelos de predição
- Melhorar a calibração de probabilidades
- Adicionar novos algoritmos de RL

### 📚 Melhorar os Notebooks
- Corrigir erros didáticos
- Adicionar explicações mais claras
- Traduzir para inglês

## Setup para Desenvolvimento

```bash
# Clone o repo
git clone https://github.com/ahirtonlopes/copa-ia.git
cd copa-ia

# Instale com dependências de dev
make install

# Rode os testes
make test

# Verifique o estilo de código
make lint
```

## Fluxo de Contribuição

1. **Fork** o repositório
2. **Crie uma branch** descritiva:
   ```bash
   git checkout -b feat/meu-novo-modelo
   # ou
   git checkout -b fix/corrige-calculo-poisson
   ```
3. **Faça suas mudanças** e escreva testes
4. **Rode os testes**: `make test`
5. **Rode o linter**: `make lint`
6. **Commit** com mensagem clara:
   ```
   feat: adiciona modelo Dixon-Coles completo com correlação de baixos scores
   fix: corrige cálculo de probabilidade em mata-mata
   docs: adiciona explicação de Monte Carlo no notebook 07
   ```
7. **Push** e abra um **Pull Request**

## Padrões de Código

- Python 3.11+
- Formatação: `ruff format` (Black-compatible)
- Linting: `ruff check`
- Docstrings em português, código em inglês
- Testes obrigatórios para novos modelos

## Código de Conduta

Este projeto segue o [Contributor Covenant](https://www.contributor-covenant.org/).
Seja respeitoso, construtivo e inclusivo.

---

Dúvidas? Abra uma issue ou entre em contato: **ahirtonlopes@gmail.com**
