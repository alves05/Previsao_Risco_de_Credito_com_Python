import os
import pickle
from datetime import datetime

import pandas as pd
import streamlit as st
from fpdf import FPDF
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def modelo_ml():
    """Carrega o modelo de Machine Learning para classificação de clientes que paga ou não o empréstimo."""
    modelo = pickle.load(open('./modelos/random_forest_credit.sav', 'rb'))
    return modelo


def modelo_grau_risco():
    """Carrega o modelo de machine learning para classificar o grau de risco do emprestimo."""
    modelo = pickle.load(open('./modelos/modelo_grau_risco.sav', 'rb'))
    return modelo


def base_dados() -> pd.DataFrame:
    """Carrega a base de dados para previsão de clientes que paga ou não."""
    dados = pd.read_csv('./dados/credit_risk_balanceado.csv')
    dados = dados.drop('loan_status', axis=True)
    return dados


def base_dados_grau_risco() -> pd.DataFrame:
    """Carrega a base de dados para previsão de grau de risco do empréstimo."""
    dados = pd.read_csv('./dados/credit_risk_balanceado_grau_risco.csv')
    dados = dados.drop(['loan_status', 'loan_grade'], axis=True)
    return dados


def processamento_base(dados: pd.DataFrame) -> list[list[list]]:
    """Processa a base de dados para utilização no modelo de previsão de cliente que paga ou não."""
    onehotencoder = ColumnTransformer(
        transformers=[('OneHot', OneHotEncoder(), [0, 2, 4, 5, 9, 10])],
        remainder='passthrough',
    )
    dados_processados = onehotencoder.fit_transform(dados).toarray()
    scaler = StandardScaler()
    dados_processados = scaler.fit_transform(dados_processados)
    dados_processados = dados_processados[-1]
    dados_processados = dados_processados.reshape(1, -1)
    return dados_processados


def processando_base_grau_risco(dados: pd.DataFrame) -> list[list[list]]:
    """Processa a base de dados para utilização no modelo de previsão de grau de risco do empréstimo."""
    onehotencoder = ColumnTransformer(
        transformers=[('OneHot', OneHotEncoder(), [0, 2, 4, 8])],
        remainder='passthrough',
    )
    dados_processados = onehotencoder.fit_transform(dados).toarray()
    scaler = StandardScaler()
    dados_processados = scaler.fit_transform(dados_processados)
    dados_processados = dados_processados[-1]
    dados_processados = dados_processados.reshape(1, -1)
    return dados_processados


def previsao(
    idade: int,
    renda: float,
    imovel: str,
    tempo_trabalho: float,
    intencao: str,
    grau: str,
    valor_emprestimo: float,
    taxa_emprestimo: float,
    taxa_rendimento: float,
    historico_inad: str,
    historico_credito: int,
) -> bool:
    """Faz a previsão do risco de crédito."""
    nova_linha = {
        'person_age': [idade],
        'person_income': [renda],
        'person_home_ownership': [imovel],
        'person_emp_length': [tempo_trabalho],
        'loan_intent': [intencao],
        'loan_grade': [grau],
        'loan_amnt': [valor_emprestimo],
        'loan_int_rate': [taxa_emprestimo],
        'loan_percent_income': [taxa_rendimento],
        'cb_person_default_on_file': [historico_inad],
        'cb_person_cred_hist_length': [historico_credito],
    }
    nova_linha = pd.DataFrame(nova_linha)
    dados = base_dados()
    dados = pd.concat([dados, nova_linha], ignore_index=True)

    modelo = modelo_ml()
    dados_previsao = processamento_base(dados)
    return modelo.predict(dados_previsao)[0]


def previsao_grau_risco(
    idade: int,
    renda: float,
    imovel: str,
    tempo_trabalho: float,
    intencao: str,
    valor_emprestimo: float,
    taxa_emprestimo: float,
    taxa_rendimento: float,
    historico_inad: str,
    historico_credito: int,
) -> bool:
    """Faz a previsão do grau de risco do empréstimo."""
    nova_linha = {
        'person_age': [idade],
        'person_income': [renda],
        'person_home_ownership': [imovel],
        'person_emp_length': [tempo_trabalho],
        'loan_intent': [intencao],
        'loan_amnt': [valor_emprestimo],
        'loan_int_rate': [taxa_emprestimo],
        'loan_percent_income': [taxa_rendimento],
        'cb_person_default_on_file': [historico_inad],
        'cb_person_cred_hist_length': [historico_credito],
    }
    nova_linha = pd.DataFrame(nova_linha)
    dados = base_dados_grau_risco()
    dados = pd.concat([dados, nova_linha], ignore_index=True)

    modelo = modelo_grau_risco()
    dados_previsao = processando_base_grau_risco(dados)
    return modelo.predict(dados_previsao)[0]


def calcula_idade(data_nascimento: datetime) -> int:
    """Calcula a idade a partir da data de nascimento fornecida."""
    data_atual = datetime.now()
    idade = (
        data_atual.year
        - data_nascimento.year
        - (
            (data_atual.month, data_atual.day)
            < (data_nascimento.month, data_nascimento.day)
        )
    )
    return idade


def idade_ano(idade: int) -> datetime:
    """Calcula o ano de nascimento conforme a idade informada."""
    data_atual = datetime.now()
    ano = data_atual.year - idade
    return ano


def financiamento_price(debito: float, parcelas: int, juros: float) -> float:
    """Função que calcula o valor das parcelas do empréstimo no metodo Price."""
    return (debito * juros) / (1 - (1 + juros) ** -parcelas)


class PDF(FPDF):
    """Classe personalizada para substituir os métodos de cabeçalho, subtitulo e rodapé."""

    def __init__(self):
        super().__init__()

    def header(self):
        self.set_font('Arial', 'B', 18)
        self.cell(0, 10, 'Banco & Bank .inc', 1, 1, 'C')

    def subtitle(self, subtitulo: str):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, subtitulo, 1, 1, 'C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', '', 11)
        self.cell(
            0, 10, str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), 1, 0, 'C'
        )


def gera_pdf(dados: pd.DataFrame, colunas: list, termo: str) -> None:
    """Cria um arquivo no formato PDF."""
    pdf = PDF()
    pdf.add_page()
    pdf.subtitle('Proposta de Empréstimo')

    pdf.set_font('Arial', '', 12)
    for linha in range(0, len(dados)):
        pdf.cell(
            w=95,
            h=10,
            txt=dados[colunas[0]].iloc[linha],
            border=1,
            ln=0,
            align='L',
        )
        pdf.cell(
            w=95,
            h=10,
            txt=str(dados[colunas[1]].iloc[linha]),
            border=1,
            ln=1,
            align='L',
        )

    pdf.ln(8)
    pdf.multi_cell(w=0, h=5, txt=termo, align='L')

    return pdf.output(f'./.relatorio/relatorio.pdf', 'f')


def main():
    st.set_page_config(page_title='Banco&Bank', page_icon='💲')
    st.image('./img/logos/logo.png', use_column_width=False)
    st.header('', divider='blue')
    st.markdown(
        "<h2 style='text-align: center; font-family: Verdana'>CreditInspector ML</h2>",
        unsafe_allow_html=True,
    )
    st.header('', divider='blue')
    st.caption('Versão 1.0')

    # Nome
    nome = st.text_input('Informe o Nome Completo:', max_chars=50).upper()

    # documento
    cpf = st.text_input('Informe o CPF (apenas números):', max_chars=11)

    # Idade
    data_nascimento = st.date_input(
        'Data de Nascimento:',
        value=None,
        min_value=datetime(idade_ano(60), 12, 31),
        max_value=datetime(idade_ano(20), datetime.now().month, datetime.now().day),
        format='DD/MM/YYYY',
    )
    st.caption('Idade mínima para solicitar empréstimo é de 20 anos.')
    if data_nascimento is not None:
        idade = calcula_idade(data_nascimento)
        st.write(str(idade), 'anos')
    else:
        st.info('Insira a data de nascimento do cliente.', icon='❕')
        idade = 0

    # Renda
    renda_mensal = st.number_input('Renda Mensal:', value=0.00)
    renda = float(renda_mensal * 12)
    if renda == 0.00:
        st.info('Informe o valor da renda mensal diferente de 0.00.', icon='❕')
    else:
        pass

    # Tipo de residência
    tipo_imovel = st.selectbox(
        'Tipo de Residência:', ['Própria', 'Financiada', 'Alugada', 'Outros']
    )
    if tipo_imovel == 'Própria':
        imovel = 'OWN'
    elif tipo_imovel == 'Financiada':
        imovel = 'MORTGAGE'
    elif tipo_imovel == 'Alugada':
        imovel = 'RENT'
    else:
        imovel = 'OTHER'

    # Tempo de emprego
    tempo_trabalho = st.slider('Tempo de Emprego (em anos):', 0, 30, 1)

    # Finalidade do empréstimo
    tipo_intencao = st.selectbox(
        'Finalidade do Empréstimo:',
        [
            'Educação',
            'Despesas Médicas',
            'Empréstimo Empresarial',
            'Pessoal',
            'Reforma de Propriedade',
            'Negociação de Dívidas',
        ],
    )
    if tipo_intencao == 'Educação':
        intencao = 'EDUCATION'
    elif tipo_intencao == 'Despesas Médicas':
        intencao = 'MEDICAL'
    elif tipo_intencao == 'Empréstimo Empresarial':
        intencao = 'VENTURE'
    elif tipo_intencao == 'Empréstimo Pessoal':
        intencao = 'PERSONAL'
    elif tipo_intencao == 'Reforma de Propriedade':
        intencao = 'HOMEIMPROVEMENT'
    else:
        intencao = 'DEBTCONSOLIDATION'

    # Valor do empréstimo
    valor_emprestimo = st.number_input('Valor do Empréstimo:', value=0.00)
    if valor_emprestimo <= 0.00:
        st.info('Informe o valor do empréstimo diferente de 0.00.', icon='❕')
    else:
        pass

    # Taxa de juros
    taxa_emprestimo = st.slider(
        'Taxa de Juros Mensal do Empréstimo (%):', 0.45, 1.93, 0.5
    )

    # Total de parcelas
    quantidade_parcelas = st.selectbox(
        'Quantidade de Parcelas: ',
        ['1x', '2x', '3x', '4x', '5x', '6x', '12x', '24x', '36x', '48x', '60x', '72x'],
    )
    if quantidade_parcelas == '1x':
        parcelas = 1
    elif quantidade_parcelas == '2x':
        parcelas = 2
    elif quantidade_parcelas == '3x':
        parcelas = 3
    elif quantidade_parcelas == '4x':
        parcelas = 4
    elif quantidade_parcelas == '5x':
        parcelas = 5
    elif quantidade_parcelas == '6x':
        parcelas = 6
    elif quantidade_parcelas == '12x':
        parcelas = 12
    elif quantidade_parcelas == '24x':
        parcelas = 24
    elif quantidade_parcelas == '36x':
        parcelas = 36
    elif quantidade_parcelas == '48x':
        parcelas = 48
    elif quantidade_parcelas == '60x':
        parcelas = 60
    else:
        parcelas = 72

    valor_parcelas = financiamento_price(
        valor_emprestimo, parcelas, (taxa_emprestimo / 100)
    )
    taxa_emprestimo_anual = round((taxa_emprestimo * 12), 2)

    # Taxa de renda para o emprestimo
    try:
        if renda != 0:
            taxa_rendimento_automatica = round(
                (valor_parcelas / renda_mensal) * 100, 2
            )
            taxa_rendimento_automatica_formatada = '{:.2f}%'.format(
                taxa_rendimento_automatica
            )
            if taxa_rendimento_automatica != 0:
                st.write(
                    'Comprometimento da Renda com Empréstimo:',
                    str(taxa_rendimento_automatica_formatada),
                )
            else:
                raise ZeroDivisionError(
                    'O valor do Empréstimo não pode ser 0.00.'
                )
        else:
            raise ZeroDivisionError(
                'O valor da Renda Mensal não pode ser 0.00.'
            )
    except ZeroDivisionError as e:
        st.info(e, icon='⚠')
        taxa_rendimento_automatica = 1

    try:
        if nome and cpf is not None:
            valor_parcelas_formatado = (
                '{:,.2f}'.format(valor_parcelas)
                .replace(',', ' ')
                .replace('.', ',')
                .replace(' ', '.')
            )
            st.write('Valor das Parcelas:', str(valor_parcelas_formatado))
        else:
            raise ValueError('Nome e CPF devem ser informados.')
    except ValueError as e:
        st.info(e, icon='⚠')
        nome = None
        cpf = None

    # Histórico de inadimplência
    tipo_inadimplencia = st.selectbox(
        'Histórico de Inadimplência:', ['SIM', 'NÃO']
    )
    if tipo_inadimplencia == 'SIM':
        historico_inadimplencia = 'Y'
    else:
        historico_inadimplencia = 'N'

    # Histórico de crédito
    historico_credito = st.slider(
        'Histórico de Crédito (em anos):',
        2,
        30,
        2,
    )

    # Grau de risco do empréstimo
    grau = ''
    try:
        if idade is not None and renda and valor_emprestimo is not 0:
            grau = previsao_grau_risco(
                idade,
                renda,
                imovel,
                tempo_trabalho,
                intencao,
                valor_emprestimo,
                taxa_emprestimo_anual,
                taxa_rendimento_automatica,
                historico_inadimplencia,
                historico_credito,
            )

            if grau == 'A':
                st.write('Risco do Empréstimo: Muito Baixo')
            elif grau == 'B':
                st.write('Risco do Empréstimo: Baixo')
            elif grau == 'C':
                st.write('Risco do Empréstimo: Médio Baixo')
            elif grau == 'D':
                st.write('Risco do Empréstimo: Médio')
            elif grau == 'E':
                st.write('Risco do Empréstimo: Médio Alto')
            elif grau == 'F':
                st.write('Risco do Empréstimo: Alto')
            else:
                st.write('Risco do Empréstimo: Muito Alto')

        else:
            raise ValueError('Verifique os dados informados.')
    except ValueError as e:
        st.info(e, icon='❕')

    if grau == 'A':
        tipo_grau = 'Muito Baixo'
    elif grau == 'B':
        tipo_grau = 'Baixo'
    elif grau == 'C':
        tipo_grau = 'Médio Baixo'
    elif grau == 'D':
        tipo_grau = 'Médio'
    elif grau == 'E':
        tipo_grau = 'Médio Alto'
    elif grau == 'F':
        tipo_grau = 'Alto'
    else:
        tipo_grau = 'Muito Alto'

    if st.button('ANALISAR CRÉDITO'):
        try:
            if (
                nome
                and cpf is not idade
                and renda
                and valor_emprestimo is not 0
            ):
                if 20 <= idade <= 60 and taxa_rendimento_automatica < 40:
                    resultado = previsao(
                        idade,
                        renda,
                        imovel,
                        tempo_trabalho,
                        intencao,
                        grau,
                        valor_emprestimo,
                        taxa_emprestimo_anual,
                        taxa_rendimento_automatica,
                        historico_inadimplencia,
                        historico_credito,
                    )

                    if resultado != 1:
                        valor = valor_emprestimo
                        valor_formatado = (
                            '{:,.2f}'.format(valor)
                            .replace(',', ' ')
                            .replace('.', ',')
                            .replace(' ', '.')
                        )
                        mensagem = (
                            f'Linha de crédito de {valor_formatado} liberada.'
                        )
                        st.success(mensagem, icon='✔')

                        st.write('Proposta de Empréstimo:')
                        ficha = {
                            'Nome do Cliente': nome,
                            'CPF': cpf,
                            'Idade': idade,
                            'Renda Mensal': renda_mensal,
                            'Tipo de Residência': tipo_imovel,
                            'Tempo de Emprego': tempo_trabalho,
                            'Finalidade do Empréstimo': tipo_intencao,
                            'Grau de Empréstimo': tipo_grau,
                            'Valor do Emprestimo': valor_emprestimo,
                            'Total de Parcelas': parcelas,
                            'Taxa de Juros Mensal (%)': taxa_emprestimo,
                            'Valor das Parcelas': valor_parcelas_formatado,
                            'Percentual de Renda p/ Empréstimo': taxa_rendimento_automatica_formatada,
                            'Histórico de Inadimplência': tipo_inadimplencia,
                            'Histórico de Crédito (em anos)': historico_credito,
                        }

                        st.dataframe(ficha, width=500, height=560)

                        relatorio = pd.DataFrame(ficha, index=[0])
                        relatorio = relatorio.transpose()
                        relatorio = relatorio.reset_index()

                        termo = f'Eu {nome} inscrito no CPF sob número {cpf}.\nDeclaro que as informações acima prestadas são verdadeiras, e assumo a inteira responsabilidade pelas mesmas.'

                        gera_pdf(relatorio, ['index', 0], termo)

                        with open('./.relatorio/relatorio.pdf', 'rb') as f:
                            st.download_button(
                                'BAIXAR PROPOSTA', f, 'relatorio.pdf'
                            )

                        # removendo arquivo
                        os.remove('./.relatorio/relatorio.pdf')

                    else:
                        st.warning(
                            'No momento não há linha de crédito disponível para o cliente.',
                            icon='❗',
                        )
                else:
                    st.warning(
                        'No momento não há linha de crédito disponível para o cliente.',
                        icon='❗',
                    )
            else:
                raise ValueError(
                    'Verifique se todos os dados foram informados.'
                )
        except ValueError as e:
            st.error(e, icon='🚨')


if __name__ == '__main__':
    main()
