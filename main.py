# Importando as bibliotecas
import pandas as pd
from pycaret.classification import *
import numpy as np
from sklearn.metrics import confusion_matrix, auc, precision_recall_curve
from sklearn.model_selection import train_test_split

# Funções para métricas customizadas
def top_k_accuracy(y_true, y_pred, k_percent=0.05):
    k = int(len(y_true) * k_percent)
    y_pred_proba = y_pred[:, 1]
    top_k_indices = np.argsort(y_pred_proba)[::-1][:k]
    correct_top_k = np.sum(y_true[top_k_indices] == 1)
    return correct_top_k / k

def bottom_k_accuracy(y_true, y_pred, k_percent=0.05):
    k = int(len(y_true) * k_percent)
    y_pred_proba = y_pred[:, 1]
    bottom_k_indices = np.argsort(y_pred_proba)[:k]
    correct_bottom_k = np.sum(y_true[bottom_k_indices] == 0)
    return correct_bottom_k / k

def npv(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fn)

def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)



# Função para executar a análise
def run_analysis(selected_columns, frac=1.0):
    # Passo 1: Carregar os dados
    file_path = 'data/DNSP_DOI_2012_2017.csv'
    data = pd.read_csv(file_path, sep=';', encoding='iso-8859-1')
    data = data[selected_columns]

    # Tratamento dos dados
    data['ANOBITO'] = data['ANOBITO'].fillna(1).apply(lambda x: 2 if x != 1 else 1)
    data = data.dropna()

    # Colunas para remover registros inválidos
    cols_to_filter = ['IDANOMAL', 'GRAVIDEZ', 'PARTO', 'ESCMAE2010', 'CONSULTAS']
    cols_to_filter = [col for col in cols_to_filter if col in data.columns]

    if 'GESTACAO' in data.columns:
        data = data[~data['GESTACAO'].isin([1, 6])]

    if cols_to_filter:
        data = data[~data[cols_to_filter].isin([9]).any(axis=1)]

    # Amostragem se necessário
    if frac < 1.0:
        data = data.sample(frac=frac, random_state=123)

    # Divisão em treino e teste
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=123)

    # Passo 2: Configuração do PyCaret
    clf_setup = setup(
    data=train_data,
    test_data=test_data,
    target='ANOBITO',
    session_id=123,
    verbose=False,
    fold=10
)

    # Passo 3: Lista de modelos a serem treinados
    models = ['lr', 'mlp', 'xgboost', 'lightgbm', 'catboost']
    results = []

    for model_name in models:
        print(f"Treinando e avaliando o modelo: {model_name}")
        model = create_model(model_name)
        tuned_model = tune_model(model, optimize='AUC', n_iter=10)
        final_model = finalize_model(tuned_model)

        # Fazendo previsões no conjunto de teste
        predictions = predict_model(final_model, data=test_data)

        if 'prediction_label' in predictions.columns and 'prediction_score' in predictions.columns:
            y_pred_proba = predictions[['prediction_label', 'prediction_score']].values
            y_pred = predictions['prediction_label'].values
        elif 'prediction' in predictions.columns and 'prediction_score' in predictions.columns:
            y_pred_proba = predictions[['prediction', 'prediction_score']].values
            y_pred = predictions['prediction'].values
        else:
            raise KeyError("Colunas de previsões não encontradas no DataFrame.")

        y_true = test_data['ANOBITO'].values

        # Calculando métricas customizadas
        top_5_accuracy = top_k_accuracy(y_true, y_pred_proba, k_percent=0.05)
        bottom_5_accuracy = bottom_k_accuracy(y_true, y_pred_proba, k_percent=0.05)
        npv_value = npv(y_true, y_pred)
        specificity_value = specificity(y_true, y_pred)
        auprc_value = auprc(y_true, y_pred_proba)

        # Métricas padrão do PyCaret
        metrics = pull()
        auc_value = metrics.loc[0, 'AUC']
        precision = metrics.loc[0, 'Prec.']
        recall = metrics.loc[0, 'Recall']
        f1 = metrics.loc[0, 'F1']

        # Salvando resultados
        results.append({
            'Algorithm': model_name,
            'AUC': auc_value,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'Top 5%': top_5_accuracy,
            'Bottom 5%': bottom_5_accuracy,
            'NPV': npv_value,
            'Specificity': specificity_value,
            'AUPRC': auprc_value,
            'Final Model': final_model
        })

    results_df = pd.DataFrame(results)
    print("Métricas de Desempenho Final:")
    print(results_df)

    return results_df

# Conjuntos de colunas para as análises
first_columns = [
    'ANONASC', 'ESFERA', 'LOCNASC', 'IDADEMAE', 'SEXO_DN', 'APGAR1', 'APGAR5', 'PESO',
    'IDANOMAL', 'GRAVIDEZ', 'PARTO', 'CONSULTAS', 'GESTACAO', 'ESCMAE2010', 'RACACORMAE',
    'CONSPRENAT', 'QTDFILVIVO', 'QTDFILMORT', 'QTDGESTANT', 'QTDPARTNOR', 'QTDPARTCES',
    'MESPRENAT', 'TPAPRESENT', 'STTRABPART', 'TPNASCASSI', 'ANOBITO'
]

second_columns = ['ANONASC', 'IDADEMAE', 'LOCNASC', 'PARTO', 'PESO', 'GESTACAO', 'ANOBITO']

third_columns = ['ANONASC', 'IDADEMAE', 'LOCNASC', 'PARTO', 'PESO', 'GESTACAO', 'APGAR1', 'APGAR5', 'IDANOMAL', 'ANOBITO']

# Executando a análise para cada conjunto de colunas
print("🏁 Primeira Análise:")
results_first = run_analysis(first_columns, frac=1)

print("\n🏁 Segunda Análise:")
results_second = run_analysis(second_columns, frac=1)

print("\n🏁 Terceira Análise:")
results_third = run_analysis(third_columns, frac=1)
