"""
CopIA Score Predictor — Mapeamento de nomes de seleções
Traduz entre o nome canônico em português (usado no schedule e UI)
e o nome em inglês usado no dataset histórico (martj42/international_results).
"""

# PT-BR → EN  (chave = nome usado no schedule/UI, valor = nome no dataset)
PT_TO_EN: dict[str, str] = {
    "Argentina":          "Argentina",
    "Brasil":             "Brazil",
    "França":             "France",
    "Espanha":            "Spain",
    "Alemanha":           "Germany",
    "Inglaterra":         "England",
    "Portugal":           "Portugal",
    "Países Baixos":      "Netherlands",
    "Bélgica":            "Belgium",
    "Colômbia":           "Colombia",
    "Marrocos":           "Morocco",
    "Croácia":            "Croatia",
    "Japão":              "Japan",
    "Uruguai":            "Uruguay",
    "Senegal":            "Senegal",
    "Coreia do Sul":      "South Korea",
    "Estados Unidos":     "United States",
    "México":             "Mexico",
    "Canadá":             "Canada",
    "Chile":              "Chile",
    "Peru":               "Peru",
    "Equador":            "Ecuador",
    "Venezuela":          "Venezuela",
    "Bolívia":            "Bolivia",
    "Panamá":             "Panama",
    "Costa Rica":         "Costa Rica",
    "Jamaica":            "Jamaica",
    "Honduras":           "Honduras",
    "Nigéria":            "Nigeria",
    "Costa do Marfim":    "Ivory Coast",
    "Camarões":           "Cameroon",
    "Gana":               "Ghana",
    "Senegal":            "Senegal",
    "Tunísia":            "Tunisia",
    "Argélia":            "Algeria",
    "Egito":              "Egypt",
    "África do Sul":      "South Africa",
    "Zâmbia":             "Zambia",
    "Congo":              "DR Congo",
    "Turquia":            "Turkey",
    "Sérvia":             "Serbia",
    "Romênia":            "Romania",
    "Eslováquia":         "Slovakia",
    "Eslovênia":          "Slovenia",
    "Hungria":            "Hungary",
    "Dinamarca":          "Denmark",
    "Suécia":             "Sweden",
    "Noruega":            "Norway",
    "Suíça":              "Switzerland",
    "Áustria":            "Austria",
    "Polônia":            "Poland",
    "República Tcheca":   "Czech Republic",
    "Escócia":            "Scotland",
    "Irlanda":            "Republic of Ireland",
    "Grécia":             "Greece",
    "Ucrânia":            "Ukraine",
    "Rússia":             "Russia",
    "Arábia Saudita":     "Saudi Arabia",
    "Irã":                "Iran",
    "Iraque":             "Iraq",
    "Austrália":          "Australia",
    "Nova Zelândia":      "New Zealand",
    "Israel":             "Israel",
    "República Centro-Africana": "Central African Republic",
    "Itália":             "Italy",
    "Bósnia":             "Bosnia and Herzegovina",
    "Albânia":            "Albania",
    "Geórgia":            "Georgia",
    # Copa 2026 grupos menos comuns
    "Iêmen":              "Yemen",
    "Chile Alt":          "Chile",
}

# Mapa inverso: EN → PT-BR
EN_TO_PT: dict[str, str] = {v: k for k, v in PT_TO_EN.items()}


def to_english(name: str) -> str:
    """Converte nome PT-BR → inglês (dataset). Retorna original se não encontrar."""
    return PT_TO_EN.get(name, name)


def to_portuguese(name: str) -> str:
    """Converte nome inglês (dataset) → PT-BR (UI). Retorna original se não encontrar."""
    return EN_TO_PT.get(name, name)


def normalize_for_model(name: str, model_teams: set) -> str:
    """
    Retorna o nome que o modelo conhece.
    Tenta: original → inglês → sem acento.
    """
    if name in model_teams:
        return name
    en = to_english(name)
    if en in model_teams:
        return en
    return name  # fallback: retorna como está (modelo usará default)
