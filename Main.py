import os
import joblib
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# utilitarios
CSV_PATH = "transito_sorocaba.csv"  # altere se necessário
MODEL_PATH = "rf_congestion_model.joblib"
ENCODERS_PATH = "encoders.joblib"


def fit_label_encoder(col_values):
    le = LabelEncoder()
    le.fit(col_values.astype(str))
    # cria mapa rápido
    label_map = {v: i for i, v in enumerate(le.classes_)}
    return le, label_map


def encode_with_fallback(label_map, le, value):
    s = str(value)
    if s in label_map:
        return label_map[s]
    else:
        # fallback: new index
        return len(label_map)

# Dados
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Arquivo CSV não encontrado em: {CSV_PATH}")

print("Carregando dados...")
df = pd.read_csv(CSV_PATH)

# Verifica colunas
expected_cols = ["hora", "dia_semana", "rua_origem", "rua_destino", "fluxo", "clima", "congestionado", "destino_hospital"]
missing = [c for c in expected_cols if c not in df.columns]
if missing:
    raise ValueError(f"As seguintes colunas estão faltando no CSV: {missing}")

# tipos certos
for c in ["hora", "dia_semana", "fluxo", "clima", "congestionado"]:
    df[c] = pd.to_numeric(df[c], errors='coerce')

df = df.dropna(subset=["hora", "dia_semana", "rua_origem", "rua_destino", "fluxo", "clima", "congestionado", "destino_hospital"]) 

#encoders e mapas
le_rua_origem, map_rua_origem = fit_label_encoder(df['rua_origem'])
le_rua_destino, map_rua_destino = fit_label_encoder(df['rua_destino'])
le_hospital, map_hospital = fit_label_encoder(df['destino_hospital'])

#codificação (train)
df['rua_origem_enc'] = df['rua_origem'].astype(str).map(map_rua_origem)
df['rua_destino_enc'] = df['rua_destino'].astype(str).map(map_rua_destino)
df['hospital_enc'] = df['destino_hospital'].astype(str).map(map_hospital)

#Matrix Blue
X = df[['hora', 'dia_semana', 'fluxo', 'clima', 'rua_origem_enc', 'rua_destino_enc', 'hospital_enc']]
Y = df['congestionado'].astype(int)

# Aqui treinei o modelo XD
print("Treinando modelo...")
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Acurácia no conjunto de teste: {acc:.4f}")
print("Relatório de classificação:\n", classification_report(y_test, y_pred))

# Salvando
joblib.dump(model, MODEL_PATH)
joblib.dump({
    'le_rua_origem': le_rua_origem,
    'map_rua_origem': map_rua_origem,
    'le_rua_destino': le_rua_destino,
    'map_rua_destino': map_rua_destino,
    'le_hospital': le_hospital,
    'map_hospital': map_hospital
}, ENCODERS_PATH)
print(f"Modelo salvo em: {MODEL_PATH}\nEncoders salvos em: {ENCODERS_PATH}")


#aqui ta criando o grafo
G = nx.Graph()
ruas = ['A','B','C','D','E','F']
hospitais = ['H1', 'H2']

for r in ruas + hospitais:
    G.add_node(r)

# Adiciona arestas
G.add_edge('A','B', base_weight=1)
G.add_edge('A','C', base_weight=2)
G.add_edge('B','C', base_weight=1.5)
G.add_edge('B','D', base_weight=2)
G.add_edge('C','D', base_weight=1)
G.add_edge('D','E', base_weight=2)
G.add_edge('E','H1', base_weight=3)
G.add_edge('D','H2', base_weight=2.5)
G.add_edge('E','F', base_weight=1.5)
G.add_edge('F','H2', base_weight=2)


def peso_ajustado(origem, destino, base_dist, hora, dia, fluxo_est, clima, hospital_dest,
                   model=model, encoders_path=ENCODERS_PATH):

    # carrega encoders
    enc = joblib.load(encoders_path)
    map_o = enc['map_rua_origem']
    map_d = enc['map_rua_destino']
    map_h = enc['map_hospital']

    o_enc = encode_with_fallback(map_o, enc['le_rua_origem'], origem)
    d_enc = encode_with_fallback(map_d, enc['le_rua_destino'], destino)
    h_enc = encode_with_fallback(map_h, enc['le_hospital'], hospital_dest)

    x = pd.DataFrame([[hora, dia, fluxo_est, clima, o_enc, d_enc, h_enc]],
                     columns=['hora', 'dia_semana', 'fluxo', 'clima', 'rua_origem_enc', 'rua_destino_enc', 'hospital_enc'])

    # Probabilidade de congestionar
    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(x)[0][1]  # probabilidade da classe 1 (congestionado)
    else:
        prob = float(model.predict(x)[0])

    # fator: mínimo 1.0, máximo 1.8
    fator = 1.0 + 0.8 * prob
    return base_dist * fator


#aqui eo principal, focaa
def encontrar_rota(grafo, origem, hospital, hora, dia, fluxo_est, clima, model=model):
    """Ajusta os pesos das arestas com base na previsão de congestionamento e encontra a menor rota até o hospital.
    Retorna caminho (lista de nós) e peso total estimado.
    """
    # copiando  o grafo pra evitar k.o
    Gcopy = grafo.copy()

    # atualiza atributos weight e calculando apartir da base weight
    for u, v, data in Gcopy.edges(data=True):
        base = data.get('base_weight', 1.0)

        adj = peso_ajustado(u, v, base, hora, dia, fluxo_est, clima, hospital)
        Gcopy[u][v]['weight'] = adj
        Gcopy[u][v]['adjusted_weight'] = adj

    try:
        caminho = nx.shortest_path(Gcopy, origem, hospital, weight='weight')
        peso_total = nx.shortest_path_length(Gcopy, origem, hospital, weight='weight')
        return caminho, peso_total, Gcopy
    except nx.NetworkXNoPath:
        return None, float('inf'), Gcopy


#exp
if __name__ == '__main__':
    # Parâmetros que a ambulância conheceria no momento da decisão (exemplo)
    origem_atual = 'A'
    hospital_alvo = 'H1'
    hora_atual = 8
    dia_atual = 1
    fluxo_estimado = 100  # pode vir de sensores ou histórico
    clima_atual = 0  # 0=claro, 1=chuva (exemplo)

    caminho, peso, Gcalc = encontrar_rota(G, origem_atual, hospital_alvo, hora_atual, dia_atual, fluxo_estimado, clima_atual)

    if caminho:
        print(f"Melhor rota de {origem_atual} até {hospital_alvo}: {' -> '.join(caminho)} (peso estimado: {peso:.2f})")

        # desenha grafo com a rota destacada
        pos = nx.spring_layout(Gcalc, seed=42)
        plt.figure(figsize=(8,6))
        nx.draw_networkx_nodes(Gcalc, pos, node_size=600)
        nx.draw_networkx_labels(Gcalc, pos)

        # desenha arestas normais
        non_route_edges = [e for e in Gcalc.edges() if not (e[0] in caminho and e[1] in caminho and abs(caminho.index(e[0]) - caminho.index(e[1]))==1)]
        nx.draw_networkx_edges(Gcalc, pos, edgelist=non_route_edges)

        # desenha arestas da rota em destaque
        route_edges = [(caminho[i], caminho[i+1]) for i in range(len(caminho)-1)]
        nx.draw_networkx_edges(Gcalc, pos, edgelist=route_edges, width=3)

        edge_labels = {(u, v): f"{d['adjusted_weight']:.2f}" for u, v, d in Gcalc.edges(data=True)}
        nx.draw_networkx_edge_labels(Gcalc, pos, edge_labels=edge_labels)
        plt.title('Grafo de ruas com rota destacada (peso ajustado nas arestas)')
        plt.axis('off')
        plt.show()
    else:
        print("Não foi encontrada rota até o hospital com os dados fornecidos.")

    print("Pronto. Você pode ajustar os parâmetros, calibrar o fator de penalização e integrar dados em tempo real (fluxo/clima).")
#Cansadonaaaaa