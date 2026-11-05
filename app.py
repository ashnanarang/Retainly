import base64
import hashlib
from html import escape
import os
from pathlib import Path
import sqlite3
from datetime import date

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.model_selection import train_test_split

from analytics import churn_by_category
from case_store import (
    create_case,
    find_duplicate,
    get_case,
    initialize_database,
    list_cases,
    remote_storage_configured,
    update_case,
)
from column_mapper import detect_columns
from data_loader import get_basic_info, load_data
from demo_data import DEMO_ACTIONS, DEMO_REASONS, build_demo_cases, build_demo_customers
from model import eligible_feature_columns, evaluate_threshold, score_customers, train_model
from preprocessing import clean_data, convert_churn_target


REASONS = list(DEMO_REASONS)
REASON_PLACEHOLDER = "— Select a reason —"
RETENTION_ACTIONS = list(DEMO_ACTIONS)
OUTCOMES = ["Pending", "Saved", "Still leaving"]
CANCELLATION_STATUSES = ["Not cancelled", "Cancelled"]
RETAINLY_SYMBOL_PATH = Path(__file__).parent / "assets" / "retainly_symbol.png"
RETAINLY_SYMBOL_DATA = base64.b64encode(RETAINLY_SYMBOL_PATH.read_bytes()).decode("ascii")
RETAINLY_SIDEBAR_SYMBOL_PATH = Path(__file__).parent / "assets" / "retainly_symbol_sidebar.png"
RETAINLY_SIDEBAR_SYMBOL_DATA = base64.b64encode(RETAINLY_SIDEBAR_SYMBOL_PATH.read_bytes()).decode("ascii")

st.set_page_config(
    page_title="Customer Intelligence & Retention Desk",
    page_icon="C",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
      --ci-ivory: #FAF9FC;
      --ci-paper: #FFFFFF;
      --ci-ink: #172A43;
      --ci-muted: #536579;
      --ci-line: #E2E8ED;
      --ci-blue: #A9D8F5;
      --ci-blue-soft: #EEF7FC;
      --ci-blush: #F7E5D7;
      --ci-mint: #CDE9DF;
      --ci-butter: #F3E7A1;
      --ci-blue-ink: #193D58;
      --ci-font-family: "Source Sans", "Segoe UI", Arial, sans-serif;
      --ci-button-blue: #A9D8F5;
      --ci-button-blue-hover: #8CC9EB;
      --ci-button-yellow: #F3E7A1;
      --ci-button-yellow-hover: #EBDD87;
      --ci-sand: #F3E7A1;
    }
    html, body, [data-testid="stAppViewContainer"] {
      background: var(--ci-ivory);
      color: var(--ci-ink);
      font-family: var(--ci-font-family);
    }
    button, input, textarea, select, [role="combobox"] {
      font-family: var(--ci-font-family);
    }
    [data-testid="stHeader"] {background: var(--ci-ivory);}
    [data-testid="stMain"] {background: var(--ci-ivory);}
    .block-container {
      max-width: 1480px;
      padding: 3.4rem 2.15rem 3.25rem;
    }
    h1, h2, h3, h4, p, label {color: var(--ci-ink);}
    h1 {font-size: 1.65rem !important; letter-spacing: -.025em; font-weight: 700 !important;}
    h2 {font-size: 1.24rem !important; letter-spacing: -.015em; font-weight: 650 !important;}
    h3 {font-size: 1.03rem !important; font-weight: 650 !important;}
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
      color: var(--ci-muted);
    }
    [data-testid="stSidebar"] {
      background: #F3F6F7;
      border-right: 1px solid var(--ci-line);
      color: var(--ci-ink) !important;
    }
    [data-testid="stSidebarContent"] {
      overflow-y: auto;
      overscroll-behavior: contain;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] h6,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
      color: var(--ci-ink) !important;
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
      color: var(--ci-muted) !important;
    }
    [data-testid="stSidebar"] [data-testid="stTooltipIcon"] button,
    [data-testid="stSidebar"] [data-testid="stTooltipIcon"] svg {
      color: var(--ci-muted) !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
      background: var(--ci-paper);
      border: 1px solid var(--ci-line);
      border-radius: 14px;
      padding: .6rem;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
      background: #F6FBFE;
      border: 1px dashed #9ECDE4;
      border-radius: 10px;
      color: var(--ci-ink);
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]:hover {
      background: var(--ci-blue-soft);
      border-color: var(--ci-blue);
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
      background: #fff !important;
      border: 1px solid #BBD9E7 !important;
      color: var(--ci-ink) !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button p,
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] [data-testid="stIconMaterial"] {
      color: var(--ci-ink) !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] {
      color: var(--ci-muted);
    }
    [data-testid="stRadio"] div[role="radiogroup"] {
      display: flex;
      flex-direction: column;
      gap: .2rem;
    }
    [data-testid="stRadio"] div[role="radiogroup"] label {
      padding: .32rem .48rem;
      border-radius: 8px;
      transition: background-color .15s ease;
    }
    [data-testid="stRadio"] input[type="radio"] {accent-color: var(--ci-blue);}
    [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
      background: var(--ci-blue-soft);
      color: var(--ci-blue-ink);
      font-weight: 650;
    }
    [data-testid="stCheckbox"] label > div:first-of-type {
      background: #FFFFFF;
      border: 1px solid #AAB7BD;
    }
    [data-testid="stCheckbox"] label:has(input:checked) > div:first-of-type {
      background: var(--ci-blue);
      border-color: var(--ci-blue);
    }
    [data-testid="stCheckbox"] label:focus-within {
      outline: 3px solid var(--ci-blue);
      outline-offset: 2px;
      border-radius: 4px;
    }
    [data-testid="stRadio"] label[data-testid="stRadioOption"]:focus-within {
      outline: 3px solid var(--ci-blue);
      outline-offset: 2px;
    }
    [data-testid="stRadio"] div[role="radiogroup"] label:hover {
      background: #F0F6F8;
    }
    button:focus-visible, input:focus-visible, textarea:focus-visible {
      outline: 3px solid var(--ci-blue) !important;
      outline-offset: 2px !important;
    }
    [data-testid="stFileUploaderDropzone"]:focus-visible,
    [data-testid="stFileUploaderDropzone"] button:focus-visible {
      outline: 3px solid var(--ci-blue) !important;
      outline-offset: 2px !important;
    }
    [data-testid="stMetric"] {
      background: var(--ci-paper);
      border: 1px solid var(--ci-line);
      border-radius: 15px;
      padding: 1.05rem 1.15rem;
      box-shadow: 0 3px 12px rgba(38, 50, 56, .04);
      min-height: 112px;
    }
    [data-testid="stMetricLabel"] p {
      color: var(--ci-muted);
      font-size: .79rem;
      font-weight: 580;
      line-height: 1.3;
    }
    [data-testid="stMetricValue"] {
      color: var(--ci-ink);
      font-size: 1.65rem;
      font-weight: 670;
      letter-spacing: -.035em;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
      background: var(--ci-paper);
      border-color: var(--ci-line) !important;
      border-radius: 16px !important;
      box-shadow: 0 4px 16px rgba(38, 50, 56, .04);
    }
    [data-testid="stForm"] {
      background: var(--ci-paper);
      border-color: var(--ci-line) !important;
      border-radius: 16px;
      padding: 1.15rem;
    }
    [data-testid="stTextInputRootElement"],
    [data-testid="stTextAreaRootElement"],
    [data-testid="stNumberInputContainer"],
    [data-testid="stDateInputField"],
    [data-testid="stSelectbox"] [role="group"],
    [data-testid="stMultiSelect"] [role="group"] {
      background-color: #FFFFFF;
      border: 1px solid #D6E0E4;
      border-radius: 9px;
    }
    [data-testid="stTextInputRootElement"]:focus-within,
    [data-testid="stTextAreaRootElement"]:focus-within,
    [data-testid="stNumberInputContainer"]:focus-within,
    [data-testid="stDateInputField"]:focus-within,
    [data-testid="stSelectbox"] [role="group"]:focus-within,
    [data-testid="stMultiSelect"] [role="group"]:focus-within {
      border-color: var(--ci-blue);
      box-shadow: 0 0 0 1px var(--ci-blue);
    }
    [data-testid="stTextInputRootElement"] input,
    [data-testid="stTextAreaRootElement"] textarea,
    [data-testid="stNumberInputContainer"] input,
    [data-testid="stDateInputField"] [role="spinbutton"],
    [data-testid="stSelectbox"] [role="combobox"],
    [data-testid="stMultiSelect"] [role="combobox"] {
      color: var(--ci-ink) !important;
    }
    [data-testid="stSelectboxVirtualDropdown"],
    [data-testid="stMultiSelectVirtualDropdown"] {
      background: var(--ci-paper) !important;
      border: 1px solid var(--ci-line) !important;
      border-radius: 10px;
      box-shadow: 0 8px 24px rgba(38, 50, 56, .12);
    }
    [data-testid="stSelectboxVirtualDropdown"] [role="listbox"],
    [data-testid="stMultiSelectVirtualDropdown"] [role="listbox"] {
      background: var(--ci-paper);
      color: var(--ci-ink);
    }
    [data-testid="stSelectboxVirtualDropdown"] [role="option"],
    [data-testid="stMultiSelectVirtualDropdown"] [role="option"],
    [data-testid="stSelectboxVirtualDropdown"] [role="option"] *,
    [data-testid="stMultiSelectVirtualDropdown"] [role="option"] * {
      color: var(--ci-ink) !important;
    }
    [data-testid="stSelectboxVirtualDropdown"] [role="option"][data-focused="true"],
    [data-testid="stSelectboxVirtualDropdown"] [role="option"][data-selected="true"],
    [data-testid="stMultiSelectVirtualDropdown"] [role="option"][data-focused="true"],
    [data-testid="stMultiSelectVirtualDropdown"] [role="option"][data-selected="true"] {
      background: var(--ci-blue-soft) !important;
    }
    [data-testid="stDateInputCalendar"] {
      background: var(--ci-paper) !important;
      border: 1px solid var(--ci-line) !important;
      border-radius: 11px;
      box-shadow: 0 8px 24px rgba(38, 50, 56, .12);
      color: var(--ci-ink);
    }
    [data-testid="stDateInputCalendar"] h2,
    [data-testid="stDateInputCalendar"] button {
      color: var(--ci-ink) !important;
    }
    [data-testid="stDateInputCalendar"] th {
      color: var(--ci-muted) !important;
      font-weight: 600;
    }
    [data-testid="stDateInputCalendar"] button {
      background: transparent;
      border: 0;
      border-radius: 7px;
    }
    [data-testid="stDateInputCalendar"] [role="gridcell"] [role="button"] {
      color: var(--ci-ink) !important;
    }
    [data-testid="stDateInputCalendar"] [role="gridcell"] [role="button"][data-disabled="true"] {
      color: #65727A !important;
      opacity: 1;
    }
    [data-testid="stDateInputCalendar"] [role="gridcell"] [role="button"][data-selected="true"] {
      background: var(--ci-blue) !important;
      color: var(--ci-ink) !important;
    }
    [data-testid="stDateInputCalendar"] [role="gridcell"] [role="button"]:hover:not([data-disabled="true"]) {
      background: var(--ci-blue-soft);
    }
    [data-testid="stDateInputCalendar"] [data-focus-visible="true"] {
      outline: 2px solid var(--ci-blue-ink) !important;
      outline-offset: -2px;
    }
    [data-testid="stNumberInputField"]:disabled {
      color: #58666D !important;
      -webkit-text-fill-color: #58666D;
      opacity: 1;
    }
    [data-testid="stNumberInputStepDown"],
    [data-testid="stNumberInputStepUp"],
    [data-testid="stNumberInputStepDown"] svg,
    [data-testid="stNumberInputStepUp"] svg {
      color: var(--ci-muted) !important;
    }
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stNumberInput"] input {background-color: transparent;}
    [data-testid="stTextInput"] input::placeholder,
    [data-testid="stTextArea"] textarea::placeholder {
      color: var(--ci-muted) !important;
      opacity: 1;
    }
    [data-testid="stButton"] button, [data-testid="stDownloadButton"] button {
      border-radius: 9px;
      font-weight: 600;
      border: 1px solid #D6E0E4;
      background: var(--ci-paper) !important;
      color: var(--ci-ink) !important;
      transition: background-color .15s ease, border-color .15s ease, box-shadow .15s ease;
    }
    [data-testid="stButton"] button[kind="primary"],
    [data-testid="stButton"] button[data-testid*="primary"],
    [data-testid="stFormSubmitButton"] button[kind="primary"],
    [data-testid="stFormSubmitButton"] button[data-testid*="primary"] {
      background: var(--ci-button-blue) !important;
      border-color: #7CB9DA !important;
      color: #183B4D !important;
    }
    [data-testid="stButton"] button[data-testid*="primary"] p,
    [data-testid="stFormSubmitButton"] button[data-testid*="primary"] p {
      color: #183B4D !important;
    }
    [data-testid="stButton"] button[kind="secondary"],
    [data-testid="stButton"] button[data-testid*="secondary"] {
      background: var(--ci-button-yellow) !important;
      border-color: #D8CC83 !important;
      color: #263238 !important;
    }
    [data-testid="stButton"] button[kind="secondary"] p,
    [data-testid="stButton"] button[data-testid*="secondary"] p {
      color: #263238 !important;
    }
    [data-testid="stButton"] button[data-testid*="secondary"] p,
    [data-testid="stDownloadButton"] button p {
      color: var(--ci-ink) !important;
    }
    [data-testid="stButton"] button[kind="primary"]:hover,
    [data-testid="stButton"] button[data-testid*="primary"]:hover,
    [data-testid="stFormSubmitButton"] button[kind="primary"]:hover,
    [data-testid="stFormSubmitButton"] button[data-testid*="primary"]:hover {
      background: var(--ci-button-blue-hover) !important;
      border-color: #63A8D0 !important;
      color: #183B4D !important;
    }
    [data-testid="stButton"] button[kind="secondary"]:hover,
    [data-testid="stButton"] button[data-testid*="secondary"]:hover {
      background: var(--ci-button-yellow-hover) !important;
      border-color: #C9BA70 !important;
      color: #263238 !important;
    }
    [data-testid="stButton"] button[kind="primary"]:active,
    [data-testid="stButton"] button[data-testid*="primary"]:active,
    [data-testid="stFormSubmitButton"] button[kind="primary"]:active {
      background: #80BDE0 !important;
      border-color: #63A8D0 !important;
    }
    [data-testid="stButton"] button[kind="secondary"]:active,
    [data-testid="stButton"] button[data-testid*="secondary"]:active {
      background: #E3D58D !important;
      border-color: #C9BA70 !important;
      color: #263238 !important;
    }
    [data-testid="stButton"] button:disabled,
    [data-testid="stFormSubmitButton"] button:disabled,
    [data-testid="stDownloadButton"] button:disabled {
      background: #E8EEF1 !important;
      border-color: #C8D2D7 !important;
      color: #58666D !important;
      opacity: 1 !important;
    }
    [data-testid="stButton"] button:focus-visible,
    [data-testid="stFormSubmitButton"] button:focus-visible,
    [data-testid="stDownloadButton"] button:focus-visible {
      outline: 3px solid #234B60 !important;
      outline-offset: 2px !important;
    }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
      border: 1px solid var(--ci-line);
      border-radius: 12px;
      overflow: hidden;
      font-family: var(--ci-font-family);
    }
    [data-testid="stAlert"] {border-radius: 11px;}
    [data-testid="stAlertContainer"] {
      background: var(--ci-blue-soft) !important;
      box-sizing: border-box;
      border: 1px solid #C8E1EF !important;
      border-left: 4px solid #4C9ED0 !important;
      border-radius: 11px !important;
    }
    [data-testid="stAlertContentInfo"] {
      background: transparent !important;
      border-left: 0 !important;
      border-radius: 0 !important;
    }
    [data-testid="stAlertContentInfo"] p {color: var(--ci-blue-ink);}
    [data-testid="stAlertContentWarning"] {
      background: transparent !important;
      border-left: 0 !important;
      border-radius: 0 !important;
    }
    [data-testid="stAlertContentWarning"] p {color: #263238 !important;}
    [data-testid="stAlertContentSuccess"] {
      background: transparent !important;
      border-left: 0 !important;
      border-radius: 0 !important;
    }
    [data-testid="stAlertContentSuccess"] p {color: #36584D !important;}
    [data-testid="stAlertContentError"] {
      background: transparent !important;
      border-left: 0 !important;
      border-radius: 0 !important;
    }
    [data-testid="stAlertContentError"] p {color: #7A3030 !important;}
    .ci-brand {
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 1rem;
      padding: .35rem 0 .7rem;
      border-bottom: 1px solid var(--ci-line);
      margin-bottom: .95rem;
    }
    .ci-brand-left {
      display: flex;
      align-items: center;
      gap: .65rem;
      flex: 1 1 0;
      min-width: 0;
    }
    .ci-mark {display: block; width: 92px; flex: 0 0 92px; line-height: 0;}
    .ci-mark img {display: block; width: 100%; height: auto; object-fit: contain;}
    .ci-product {
      color: var(--ci-ink);
      font-weight: 680;
      font-size: 1rem;
      letter-spacing: -.02em;
      line-height: 1.35;
      overflow-wrap: break-word;
      word-break: normal;
    }
    .ci-product-subtitle {color: var(--ci-muted); font-size: .78rem; margin-top: .12rem;}
    .ci-storage {
      display: inline-flex; align-items: center; gap: .42rem;
      border: 1px solid #C5DFEC; background: var(--ci-blue-soft);
      color: var(--ci-blue-ink); border-radius: 999px; padding: .36rem .7rem;
      font-size: .74rem; font-weight: 600; white-space: nowrap;
    }
    .ci-storage-dot {width: 7px; height: 7px; border-radius: 50%; background: var(--ci-blue);}
    .ci-eyebrow {
      color: var(--ci-blue-ink); text-transform: uppercase; letter-spacing: .12em;
      font-size: .68rem; font-weight: 720; margin-bottom: .3rem;
    }
    .ci-page-title {font-size: 1.72rem; font-weight: 660; letter-spacing: -.04em; line-height: 1.18;}
    .ci-welcome-title {
      color: var(--ci-ink); font-family: var(--ci-font-family);
      font-size: clamp(1.55rem, 2.4vw, 2.2rem); font-weight: 700;
      line-height: 1.1; letter-spacing: -.035em; margin: .25rem 0 .65rem;
    }
    .ci-welcome-lede {
      color: #4F6068; font-size: .95rem; line-height: 1.6;
      max-width: 46rem; margin-bottom: .35rem;
    }
    .ci-welcome-art {
      display: grid; place-items: center; width: 100%; min-height: 215px;
      border: 1px solid #D4E6EE; border-radius: 18px;
      background: linear-gradient(145deg, #F1F8FC, #FBF9ED);
      padding: .8rem;
    }
    .ci-welcome-art svg {width: min(100%, 360px); height: auto;}
    .ci-step-number {
      display: grid; place-items: center; width: 2rem; height: 2rem;
      border-radius: 50%; background: var(--ci-button-blue);
      color: #183B4D; font-weight: 700; margin-bottom: .65rem;
    }
    .ci-step-title {color: var(--ci-ink); font-weight: 650; line-height: 1.4;}
    .ci-step-copy {color: var(--ci-muted); font-size: .84rem; line-height: 1.5; margin-top: .35rem;}
    .ci-welcome-card {display: flex; flex-direction: column;}
    .ci-welcome-card .ci-card-copy {min-height: 0;}
    .ci-info-card {
      min-height: 92px; border: 1px solid var(--ci-line);
      border-radius: 12px; background: var(--ci-paper);
      padding: .9rem 1rem; color: #4F6068;
      font-size: .88rem; line-height: 1.55;
    }
    .ci-info-card strong {color: var(--ci-ink);}
    .ci-page-copy {color: var(--ci-muted); margin-top: .34rem; font-size: .92rem;}
    .ci-card-title {color: var(--ci-ink); font-size: 1rem; font-weight: 650; margin-bottom: .1rem;}
    .ci-card-copy {font-size: .81rem; color: var(--ci-muted); line-height: 1.45;}
    .ci-status-note {
      border: 1px solid #C8E1EF;
      border-left: 4px solid #4C9ED0;
      background: #EEF7FC;
      border-radius: 11px;
      padding: .7rem .9rem;
      color: #193D58;
      font-size: .84rem;
      line-height: 1.5;
    }
    .ci-demo-banner {
      border: 1px solid #C8E1EF;
      border-left: 4px solid #4C9ED0;
      border-radius: 11px;
      background: var(--ci-blue-soft);
      color: var(--ci-blue-ink);
      padding: .72rem .9rem; margin: .25rem 0 1rem;
      font-size: .84rem; font-weight: 700; letter-spacing: .015em;
    }
    .ci-sidebar-label {
      color: var(--ci-muted); font-size: .68rem; font-weight: 700;
      letter-spacing: .12em; text-transform: uppercase; margin: .8rem 0 .25rem;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"] {
      background: #F3E7A1 !important;
      border-color: #D9CB78 !important;
      color: #172A43 !important;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button[data-testid*="primary"] p {
      color: #172A43 !important;
    }
    @media (max-width: 760px) {
      .block-container {padding: 3.65rem .85rem 2rem;}
      [data-testid="stHorizontalBlock"] {flex-wrap: wrap !important;}
      [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        min-width: min(100%, 155px) !important;
        flex: 1 1 155px !important;
      }
      .ci-brand {align-items: flex-start; gap: .65rem;}
      .ci-brand-left {flex-basis: 100%;}
      .ci-storage {font-size: .68rem; padding: .3rem .5rem;}
      .ci-page-title {font-size: 1.45rem;}
      [data-testid="stMetric"] {min-height: 98px; padding: .8rem;}
      [data-testid="stMetricValue"] {font-size: 1.35rem;}
    }
    @media (max-width: 480px) {
      [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        min-width: 100% !important;
        flex-basis: 100% !important;
      }
      .ci-product {font-size: .93rem;}
    }
    :root {
      --ci-shadow: 0 8px 24px rgba(48, 42, 75, .045);
    }
    html, body, button, input, textarea, select, [role="combobox"] {
      font-family: var(--ci-font-family);
    }
    [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stHeader"],
    [data-testid="stMainBlockContainer"] {background: var(--ci-ivory);}
    .block-container {max-width: 1440px;}
    [data-testid="stSidebar"] {
      background: #10233F;
      border-right: 1px solid var(--ci-line);
    }
    [data-testid="stSidebarContent"] {padding: 1.15rem .8rem 1.4rem;}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
      font-size: 1.06rem !important;
      letter-spacing: -.025em;
    }
    .ci-sidebar-label {
      color: var(--ci-muted);
      letter-spacing: .09em;
      font-size: .66rem;
      margin: 1.05rem .25rem .4rem;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [role="group"] {
      border: 1px solid var(--ci-line);
      border-radius: 10px;
      background: #FFFFFF;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] {margin: .12rem 0;}
    [data-testid="stSidebar"] [data-testid="stButton"] button {
      min-height: 2.5rem;
      justify-content: flex-start;
      border-radius: 10px !important;
      border: 1px solid transparent !important;
      padding: .5rem .65rem !important;
      font-size: .84rem;
      font-weight: 560;
      text-align: left;
      transition: background-color .18s ease, border-color .18s ease, color .18s ease;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"] {
      background: #F3E7A1 !important;
      border-color: #D9CB78 !important;
      color: #172A43 !important;
      box-shadow: none !important;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button[kind="secondary"] {
      background: transparent !important;
      color: #D7E4F0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button[kind="secondary"]:hover {
      background: #1B3655 !important;
      border-color: #33516F !important;
      color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"] p,
    [data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"] [data-testid="stIconMaterial"] {
      color: #172A43 !important;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"]:hover {
      background: #EBDD87 !important;
      border-color: #CDBD69 !important;
      color: #172A43 !important;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button p {
      color: inherit !important;
      font-weight: inherit;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button [data-testid="stIconMaterial"] {
      color: #A9D8F5 !important;
      font-size: 1.08rem;
    }
    .ci-sidebar-promo {
      margin: 1.15rem .15rem .15rem;
      padding: .95rem;
      border: 1px solid #31506E;
      border-radius: 13px;
      background: linear-gradient(135deg, #193654, #1B4261 72%, #244565);
    }
    .ci-sidebar-promo-title {color: #F5F8FC; font-size: .82rem; font-weight: 680;}
    .ci-sidebar-promo-copy {color: #CAD9E7; font-size: .73rem; line-height: 1.5; margin-top: .3rem;}
    .ci-brand {
      padding: .15rem 0 .2rem;
      gap: .75rem;
      margin-bottom: 0;
      border-bottom: 0;
    }
    .ci-product {font-size: .97rem; font-weight: 680; color: var(--ci-ink);}
    .ci-product-subtitle {font-size: .76rem; color: var(--ci-muted);}
    .ci-storage {
      background: #EEF7FC;
      border-color: #C6E0EE;
      color: #193D58;
    }
    .ci-storage-dot {background: #4C9ED0;}
    .ci-header-workspace-label {
      margin: 0 0 .3rem .1rem;
      color: var(--ci-muted);
      font-size: .66rem;
      font-weight: 650;
      letter-spacing: .08em;
      text-transform: uppercase;
    }
    .ci-header-divider {border-top: 1px solid var(--ci-line); margin: .25rem 0 .8rem;}
    .ci-eyebrow {color: #2477A8;}
    .ci-page-title {color: var(--ci-ink); font-size: 1.72rem; line-height: 1.22;}
    .ci-page-copy {color: var(--ci-muted); line-height: 1.55;}
    [data-testid="stVerticalBlockBorderWrapper"] {
      background: #FFFFFF;
      border-color: var(--ci-line) !important;
      border-radius: 14px !important;
      box-shadow: var(--ci-shadow);
    }
    [data-testid="stForm"] {
      background: #FFFFFF;
      border-color: var(--ci-line) !important;
      border-radius: 14px;
    }
    [data-testid="stMetric"] {
      min-height: 112px;
      padding: .95rem 1rem;
      border: 1px solid var(--ci-line);
      border-radius: 14px;
      box-shadow: var(--ci-shadow);
      background: #FFFFFF;
      transition: transform .18s ease, box-shadow .18s ease;
    }
    [data-testid="stMetric"]:hover {
      transform: translateY(-1px);
      box-shadow: 0 10px 24px rgba(48, 42, 75, .075);
    }
    [data-testid="stMetricLabel"] p {
      color: var(--ci-muted);
      font-size: .76rem;
      font-weight: 580;
      line-height: 1.35;
    }
    [data-testid="stMetricValue"] {
      color: var(--ci-ink);
      font-size: 1.55rem;
      font-weight: 680;
    }
    [data-testid="stTextInputRootElement"], [data-testid="stTextAreaRootElement"],
    [data-testid="stNumberInputContainer"], [data-testid="stDateInputField"],
    [data-testid="stSelectbox"] [role="group"], [data-testid="stMultiSelect"] [role="group"] {
      border-color: #E5E2EB;
      border-radius: 10px;
      min-height: 2.7rem;
    }
    [data-testid="stTextInputRootElement"]:focus-within,
    [data-testid="stTextAreaRootElement"]:focus-within,
    [data-testid="stNumberInputContainer"]:focus-within,
    [data-testid="stDateInputField"]:focus-within,
    [data-testid="stSelectbox"] [role="group"]:focus-within,
    [data-testid="stMultiSelect"] [role="group"]:focus-within {
      border-color: #4C9ED0 !important;
      box-shadow: 0 0 0 2px rgba(76, 158, 208, .18) !important;
    }
    [data-testid="stButton"] button, [data-testid="stDownloadButton"] button,
    [data-testid="stFormSubmitButton"] button {
      border: 1px solid #E5E2EB !important;
      border-radius: 10px !important;
      background: #FFFFFF !important;
      color: var(--ci-ink) !important;
      font-weight: 600;
      transition: background-color .18s ease, border-color .18s ease, box-shadow .18s ease, transform .18s ease;
    }
    [data-testid="stButton"] button[kind="primary"],
    [data-testid="stButton"] button[data-testid*="primary"],
    [data-testid="stFormSubmitButton"] button[kind="primary"],
    [data-testid="stFormSubmitButton"] button[data-testid*="primary"] {
      background: var(--ci-button-blue) !important;
      border-color: #78BBDD !important;
      color: #19344B !important;
    }
    [data-testid="stButton"] button[kind="primary"] p,
    [data-testid="stButton"] button[data-testid*="primary"] p,
    [data-testid="stFormSubmitButton"] button p {color: #25213B !important;}
    [data-testid="stButton"] button[kind="secondary"],
    [data-testid="stButton"] button[data-testid*="secondary"] {
      background: #FFFFFF !important;
      color: #51466F !important;
    }
    [data-testid="stButton"] button[kind="primary"]:hover,
    [data-testid="stFormSubmitButton"] button[kind="primary"]:hover {
      background: var(--ci-button-blue-hover) !important;
      border-color: #6EB2D7 !important;
      box-shadow: 0 4px 12px rgba(41, 106, 145, .16);
    }
    [data-testid="stButton"] button[kind="secondary"]:hover,
    [data-testid="stDownloadButton"] button:hover {
      background: #EEF7FC !important;
      border-color: #C8E1EF !important;
    }
    [data-testid="stButton"] button:disabled,
    [data-testid="stFormSubmitButton"] button:disabled,
    [data-testid="stDownloadButton"] button:disabled {
      background: #F1F0F4 !important;
      border-color: #E4E2E9 !important;
      color: #77798A !important;
      opacity: 1 !important;
    }
    button:focus-visible, input:focus-visible, textarea:focus-visible {
      outline: 3px solid rgba(44, 126, 174, .72) !important;
      outline-offset: 2px !important;
    }
    [data-testid="stFileUploader"] {
      border: 1px solid var(--ci-line);
      border-radius: 13px;
      background: #FFFFFF;
    }
    [data-testid="stFileUploaderDropzone"] {
      border-color: #C8E1EF !important;
      border-radius: 10px !important;
      background: #FAF9FC !important;
    }
    [data-testid="stFileUploaderDropzone"]:hover {background: #EEF7FC !important;}
    [data-testid="stDataFrame"], [data-testid="stTable"] {
      border: 1px solid var(--ci-line);
      border-radius: 12px;
      overflow: hidden;
      background: #FFFFFF;
      font-family: var(--ci-font-family);
    }
    [data-testid="stAlertContainer"] {
      box-sizing: border-box;
      border: 1px solid #C8E1EF !important;
      border-left: 4px solid #4C9ED0 !important;
      border-radius: 11px !important;
    }
    [data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) {
      background: #FBF7EA !important;
    }
    [data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) {
      background: #EFF7F3 !important;
    }
    [data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) {
      background: #FBF0F2 !important;
    }
    [data-testid="stAlertContentInfo"] p {color: #193D58 !important;}
    [data-testid="stAlertContentWarning"] {
      background: transparent !important;
      border-left: 0 !important;
      border-radius: 0 !important;
    }
    [data-testid="stAlertContentSuccess"] {
      background: transparent !important;
      border-left: 0 !important;
      border-radius: 0 !important;
    }
    [data-testid="stAlertContentSuccess"] p {color: #36584D !important;}
    [data-testid="stAlertContentError"] {
      background: transparent !important;
      border-left: 0 !important;
      border-radius: 0 !important;
    }
    [data-testid="stAlertContentError"] p {color: #713C4C !important;}
    [data-testid="stRadio"] input[type="radio"] {accent-color: #4C9ED0;}
    [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
      background: var(--ci-blue-soft);
      color: #193D58;
    }
    [data-testid="stCheckbox"] label:has(input:checked) > div:first-of-type {
      background: #4C9ED0;
      border-color: #4C9ED0;
    }
    [data-testid="stCheckbox"] label:focus-within,
    [data-testid="stRadio"] label:focus-within {
      outline-color: rgba(44, 126, 174, .72) !important;
    }
    .ci-demo-banner {
      background: #EEF7FC;
      border-color: #C8E1EF;
      border-left-color: #4C9ED0;
      color: #193D58;
      border-radius: 11px;
    }
    .ci-status-note {
      background: #EEF7FC;
      border-color: #C8E1EF;
      border-left-color: #4C9ED0;
      color: #193D58;
      border-radius: 11px;
    }
    .ci-welcome-title {
      color: var(--ci-ink);
      font-family: var(--ci-font-family);
      font-size: clamp(1.55rem, 2.4vw, 2.2rem);
      line-height: 1.16;
      letter-spacing: -.04em;
    }
    .ci-welcome-lede {color: #55596C; font-size: .95rem; line-height: 1.6;}
    .ci-welcome-art {
      background: linear-gradient(135deg, #EAF6FC 0%, #F4FAFD 58%, #FBF8E8 100%);
      border-color: #CFE4EF;
      border-radius: 16px;
      min-height: 205px;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:has(.ci-welcome-title) {
      background: linear-gradient(115deg, #EAF6FC 0%, #F4FAFD 58%, #FBF8E8 100%);
      border-color: #CFE4EF !important;
    }
    .ci-step-number {background: #D9EFFA; color: #193D58;}
    .ci-step-title {color: var(--ci-ink);}
    .ci-info-card {color: #55596C; border-color: var(--ci-line);}
    .ci-card-title {color: var(--ci-ink); font-weight: 650;}
    .ci-card-copy {color: var(--ci-muted);}
    .ci-sidebar-promo {display: block;}
    .ci-kpi-icon {
      display: grid;
      place-items: center;
      width: 1.9rem;
      height: 1.9rem;
      flex: 0 0 1.9rem;
      border: 1px solid #CFE4EF;
      border-radius: 9px;
      background: #EAF6FC;
      color: #2879A7;
    }
    .ci-kpi-icon svg {width: 1.05rem; height: 1.05rem;}
    .ci-kpi-card {
      min-height: 116px;
      padding: .9rem .95rem;
      border: 1px solid var(--ci-line);
      border-radius: 14px;
      background: #FFFFFF;
      box-shadow: var(--ci-shadow);
      transition: transform .18s ease, box-shadow .18s ease;
    }
    .ci-kpi-card:hover {transform: translateY(-1px); box-shadow: 0 10px 24px rgba(48, 42, 75, .075);}
    .ci-kpi-heading {display: flex; align-items: center; gap: .55rem; min-width: 0;}
    .ci-kpi-label {color: var(--ci-muted); font-size: .73rem; font-weight: 590; line-height: 1.3;}
    .ci-kpi-value {color: var(--ci-ink); font-size: 1.55rem; font-weight: 680; letter-spacing: -.035em; margin-top: .75rem;}
    [data-testid="stHorizontalBlock"]:has(.ci-kpi-card) > [data-testid="stColumn"] {margin-bottom: .45rem;}
    [data-testid="stHorizontalBlock"]:has(.ci-kpi-card) {
      flex-wrap: wrap;
      gap: .75rem;
    }
    [data-testid="stHorizontalBlock"]:has(.ci-kpi-card) > [data-testid="stColumn"] {
      min-width: 145px;
      flex: 1 1 145px;
    }
    @media (max-width: 1050px) {
      .block-container {padding-left: 1.3rem; padding-right: 1.3rem;}
      [data-testid="stHorizontalBlock"]:has(.ci-kpi-card) > [data-testid="stColumn"] {
        min-width: 185px;
        flex-basis: 185px;
      }
    }
    @media (max-width: 760px) {
      .block-container {padding: 3.5rem .9rem 2rem;}
      .ci-brand {align-items: flex-start;}
      .ci-brand-left {flex-basis: 100%;}
      .ci-storage {margin-left: 3.05rem;}
      [data-testid="stHorizontalBlock"]:has(.ci-kpi-card) > [data-testid="stColumn"] {
        min-width: min(100%, 150px);
        flex-basis: 150px;
      }
      .ci-kpi-card {min-height: 105px; padding: .8rem;}
    }
    @media (max-width: 480px) {
      [data-testid="stHorizontalBlock"]:has(.ci-kpi-card) > [data-testid="stColumn"] {
        min-width: calc(50% - .4rem) !important;
        max-width: calc(50% - .4rem) !important;
        flex: 1 1 calc(50% - .4rem) !important;
      }
    }
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after {scroll-behavior: auto !important; transition-duration: .01ms !important;}
    }
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
      background: #10233F !important;
      color: #F3F7FB !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong {
      color: #F3F7FB !important;
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
    [data-testid="stSidebar"] .ci-sidebar-label {
      color: #AFC2D5 !important;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [role="group"] {
      background: #FFFFFF !important;
      border-color: #CBD9E5 !important;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [role="combobox"],
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [role="combobox"] * {
      color: #172A43 !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"],
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
      background: #F7FBFE !important;
      color: #172A43 !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"],
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] * {
      color: #506278 !important;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"] {
      background: #F3E7A1 !important;
      border-color: #D9CB78 !important;
      color: #172A43 !important;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"] p,
    [data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"] [data-testid="stIconMaterial"] {
      color: #172A43 !important;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button[kind="secondary"] p,
    [data-testid="stSidebar"] [data-testid="stButton"] button[kind="secondary"] [data-testid="stIconMaterial"] {
      color: #D7E4F0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button:focus-visible {
      outline-color: #F3E7A1 !important;
    }
    [class*="st-key-workspace_card_"] {
      box-sizing: border-box;
      min-height: 158px;
      padding: .8rem !important;
      gap: .5rem !important;
    }
    [class*="st-key-workspace_card_"] [data-testid="stElementContainer"]:has(.ci-welcome-card) {
      margin-bottom: 0 !important;
    }
    [class*="st-key-workspace_card_"] [data-testid="stElementContainer"]:has([data-testid="stButton"]) {
      margin-top: auto !important;
    }
    [data-testid="stSidebar"] .ci-brand,
    [data-testid="stSidebar"] .ci-product {color: #F3F7FB !important;}
    [data-testid="stSidebar"] .ci-product-subtitle {color: #AFC2D5 !important;}
    .ci-sidebar-brand {
      display: flex;
      align-items: center;
      gap: .65rem;
      min-width: 0;
      padding: .25rem .1rem .8rem;
    }
    .ci-sidebar-brand .ci-mark {width: 76px; max-width: 100%; flex: 0 0 76px;}
    .ci-sidebar-brand-copy {min-width: 0;}
    .ci-sidebar-brand-name {
      color: #F3F7FB;
      font-size: 1.08rem;
      font-weight: 720;
      letter-spacing: -.025em;
      line-height: 1.15;
    }
    .ci-sidebar-brand-tagline {
      margin-top: .24rem;
      color: #C0CEDD;
      font-size: .54rem;
      font-weight: 650;
      letter-spacing: .075em;
      line-height: 1.4;
      overflow-wrap: anywhere;
    }
    .ci-sidebar-demo-badge {
      display: block;
      width: 100%;
      box-sizing: border-box;
      margin: .15rem 0 .45rem;
      padding: .55rem .65rem;
      border: 1px solid #C8E1EF;
      border-left: 4px solid #4C9ED0;
      border-radius: 9px;
      background: #F3E7A1;
      color: #1B2D46;
      font-size: .69rem;
      font-weight: 760;
      line-height: 1.35;
      letter-spacing: .015em;
      overflow-wrap: anywhere;
    }
    .ci-sidebar-demo-copy {
      margin: 0 .05rem .3rem;
      color: #B8C9D9;
      font-size: .73rem;
      line-height: 1.5;
    }
    [class*="st-key-workspace_card_"] [data-testid="stButton"] button[kind="secondary"] {
      background: #F3E7A1 !important;
      border-color: #D9CB78 !important;
      color: #172A43 !important;
      box-shadow: 0 2px 6px rgba(27, 45, 70, .06);
    }
    [class*="st-key-workspace_card_"] [data-testid="stButton"] button[kind="secondary"] p,
    [class*="st-key-workspace_card_"] [data-testid="stButton"] button[kind="secondary"] [data-testid="stIconMaterial"] {
      color: #172A43 !important;
    }
    [class*="st-key-workspace_card_"] [data-testid="stButton"] button[kind="secondary"]:hover:not(:disabled) {
      background: #EBDD87 !important;
      border-color: #CDBD69 !important;
      color: #172A43 !important;
      box-shadow: 0 4px 10px rgba(27, 45, 70, .1);
      transform: translateY(-1px);
    }
    [class*="st-key-workspace_card_"] [data-testid="stButton"] button[kind="secondary"]:focus-visible {
      outline: 3px solid #287FB2 !important;
      outline-offset: 3px !important;
    }
    [class*="st-key-workspace_card_"] [data-testid="stButton"] button[kind="secondary"]:disabled {
      background: #EEE9C9 !important;
      border-color: #DED7A8 !important;
      color: #5D5E58 !important;
      opacity: 1 !important;
    }
    [class*="st-key-workspace_card_"] [data-testid="stButton"] button[kind="secondary"]:disabled p {
      color: #5D5E58 !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:has(.ci-dashboard-art) {
      padding: 1.05rem 1.15rem;
      border-color: #C9E2F0 !important;
      border-radius: 18px !important;
      background: linear-gradient(112deg, #E6F4FB 0%, #F2F9FC 62%, #FBF8E8 100%);
      box-shadow: 0 7px 22px rgba(28, 67, 94, .06);
    }
    .ci-dashboard-title {margin: .2rem 0 .45rem; color: #172A43; font-size: clamp(1.25rem, 2vw, 1.75rem); font-weight: 700; line-height: 1.2; letter-spacing: -.025em;}
    .ci-dashboard-copy {color: #506278; font-size: .9rem; line-height: 1.55; margin: 0 0 .65rem;}
    .ci-dashboard-art {display: grid; place-items: center; min-height: 130px;}
    .ci-dashboard-art svg {width: min(100%, 290px); height: auto;}
    .ci-insight-card {
      min-height: 112px;
      padding: .9rem 1rem;
      border: 1px solid #D9E5EC;
      border-top: 3px solid #65AED6;
      border-radius: 13px;
      background: #FFFFFF;
      box-shadow: 0 4px 14px rgba(28, 67, 94, .045);
    }
    .ci-insight-title {color: #193D58; font-size: .8rem; font-weight: 700; margin-bottom: .35rem;}
    .ci-insight-copy {color: #536579; font-size: .82rem; line-height: 1.5;}
    .ci-kpi-card {border-color: #DCE7ED; box-shadow: 0 4px 14px rgba(28, 67, 94, .045);}
    .ci-kpi-icon {border-color: #D1E6F1; background: #EAF6FC; color: #2879A7;}
    .ci-kpi-value {color: #172A43;}
    .ci-kpi-card:hover {box-shadow: 0 10px 24px rgba(28, 67, 94, .09);}
    [data-testid="stVerticalBlockBorderWrapper"] {
      border-color: #DCE5EB !important;
      box-shadow: 0 4px 16px rgba(28, 54, 80, .045);
    }
    @media (max-width: 760px) {
      .ci-dashboard-art {min-height: 105px;}
    }
    html, body {background: #FAFBFD !important;}
    [data-testid="stAppViewContainer"] {
      background: linear-gradient(128deg, #FBFCFE 0%, #F5F9FC 52%, #EAF4FA 100%) !important;
      background-attachment: fixed !important;
    }
    [data-testid="stMain"], [data-testid="stHeader"], [data-testid="stMainBlockContainer"],
    .block-container {background: transparent !important;}
    .ci-sidebar-brand .ci-mark {width: 76px; max-width: 100%; flex-basis: 76px;}
    @media (max-width: 760px) {
      .ci-mark {width: 84px; flex-basis: 84px;}
      .ci-brand-left {gap: .55rem;}
    }
    @media (max-width: 480px) {
      .ci-mark {width: 76px; flex-basis: 76px;}
      .ci-product {font-size: .88rem;}
      .ci-product-subtitle {font-size: .68rem;}
      .ci-sidebar-brand {gap: .5rem;}
      .ci-sidebar-brand .ci-mark {width: 66px; flex-basis: 66px;}
      .ci-sidebar-brand-name {font-size: 1rem;}
      .ci-sidebar-brand-tagline {font-size: .5rem; letter-spacing: .055em;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def _configure_remote_case_storage():
    try:
        settings = st.secrets.get("supabase", {})
    except Exception:
        settings = {}
    if settings.get("url"):
        os.environ["SUPABASE_URL"] = str(settings["url"])
    if settings.get("service_role_key"):
        os.environ["SUPABASE_SERVICE_ROLE_KEY"] = str(settings["service_role_key"])
    if settings.get("require_remote_db", False):
        os.environ["CANCELLATION_REQUIRE_REMOTE_DB"] = "true"


_configure_remote_case_storage()

try:
    initialize_database()
except sqlite3.Error:
    st.error("Case storage could not be opened. Check the SQLite path or Supabase secrets and confirm that supabase_schema.sql has been applied.")
    st.stop()

PAGES = {
    "Welcome": (
        "Welcome",
        "An introduction to customer retention operations and insight.",
    ),
    "Overview": (
        "Retention Overview",
        "A clear view of cancellation requests, confirmed outcomes, and revenue exposure.",
    ),
    "New Cancellation Review": (
        "New cancellation review",
        "Capture the customer’s stated reason, the retention response, and the eventual outcome.",
    ),
    "Case History": (
        "Case history",
        "Search and manage review cases while preserving the original customer-stated reason.",
    ),
    "Churn Analysis": (
        "Historical churn analysis",
        "Explore historical labels, evaluate a model, and optionally score eligible active customers.",
    ),
    "Data Quality/Settings": (
        "Data quality & settings",
        "Review imported data quality, confirm column mapping, and check storage guidance.",
    ),
}


def _reset_demo_workspace():
    for key in ("demo_model", "demo_model_metrics", "demo_test_scores"):
        st.session_state.pop(key, None)
    st.session_state["workspace_page"] = "Overview"


def _persist_workspace_mode():
    if st.session_state["workspace_mode"] == "Demo workspace":
        st.query_params["ci_workspace"] = "demo"
    else:
        st.query_params.pop("ci_workspace", None)
    if "header_workspace_mode" in st.session_state:
        st.session_state["header_workspace_mode"] = st.session_state["workspace_mode"]


def _apply_header_workspace_mode():
    st.session_state["workspace_mode"] = st.session_state["header_workspace_mode"]
    _persist_workspace_mode()


def _select_page(page_name):
    st.session_state["workspace_page"] = page_name


def _reset_case_filters(key_prefix):
    st.session_state.pop(f"{key_prefix}_date", None)
    st.session_state[f"{key_prefix}_search"] = ""
    for suffix in ("reason", "outcome", "cancel", "plan", "owner"):
        st.session_state[f"{key_prefix}_{suffix}"] = "All"


query_workspace = st.query_params.get("ci_workspace")
if query_workspace in {"real", "demo"}:
    st.session_state["workspace_mode"] = (
        "Demo workspace" if query_workspace == "demo" else "Real workspace"
    )
elif st.session_state.get("workspace_mode") not in {"Real workspace", "Demo workspace"}:
    st.session_state["workspace_mode"] = "Real workspace"
if st.session_state.get("workspace_page") not in PAGES:
    st.session_state["workspace_page"] = "Welcome"
st.session_state["header_workspace_mode"] = st.session_state["workspace_mode"]

uploaded_file = None
with st.sidebar:
    st.markdown(
        f'<div class="ci-sidebar-brand"><div class="ci-mark"><img src="data:image/png;base64,{RETAINLY_SIDEBAR_SYMBOL_DATA}" alt="Retainly connected customer-retention symbol"></div><div class="ci-sidebar-brand-copy"><div class="ci-sidebar-brand-name">Retainly</div><div class="ci-sidebar-brand-tagline">TURN INSIGHTS INTO LOYALTY</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="ci-sidebar-label">Workspace</div>', unsafe_allow_html=True)
    st.selectbox(
        "Choose workspace",
        ("Real workspace", "Demo workspace"),
        key="workspace_mode",
        label_visibility="collapsed",
        on_change=_persist_workspace_mode,
    )
    is_demo = st.session_state["workspace_mode"] == "Demo workspace"
    if is_demo:
        st.markdown(
            '<div class="ci-sidebar-demo-badge" role="status">DEMO DATA — NOT REAL CUSTOMERS</div>'
            '<div class="ci-sidebar-demo-copy">Synthetic records stay in memory and never enter SQLite.</div>',
            unsafe_allow_html=True,
        )
        st.button(
            "Reset demo workspace",
            key="reset_demo_workspace",
            on_click=_reset_demo_workspace,
            help="Clear demo model results and return to the generated starting examples.",
        )
    else:
        st.markdown('<div class="ci-sidebar-label">Historical data</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload a customer CSV",
            type=["csv"],
            help="Used only for historical churn analysis and scoring. Review cases stay in SQLite and are never added to this dataset.",
            key="historical_csv",
        )
    st.markdown('<div class="ci-sidebar-label">Navigation</div>', unsafe_allow_html=True)
    navigation_labels = {"Overview": "Retention Overview"}
    navigation_icons = {
        "Welcome": ":material/waving_hand:",
        "Overview": ":material/space_dashboard:",
        "New Cancellation Review": ":material/edit_note:",
        "Case History": ":material/history:",
        "Churn Analysis": ":material/monitoring:",
        "Data Quality/Settings": ":material/tune:",
    }
    for navigation_page in PAGES:
        st.button(
            navigation_labels.get(navigation_page, navigation_page),
            key=f"navigate_{navigation_page}",
            icon=navigation_icons[navigation_page],
            type="primary" if navigation_page == st.session_state["workspace_page"] else "secondary",
            width="stretch",
            on_click=_select_page,
            args=(navigation_page,),
        )
    st.markdown(
        '<div class="ci-sidebar-promo"><div class="ci-sidebar-promo-title">A clearer view of retention</div>'
        '<div class="ci-sidebar-promo-copy">Bring customer feedback, follow-up, and outcomes together to spot useful patterns.</div></div>',
        unsafe_allow_html=True,
    )

is_demo = st.session_state["workspace_mode"] == "Demo workspace"
demo_customers = build_demo_customers() if is_demo else None
demo_cases = build_demo_cases(demo_customers) if demo_customers is not None else None
page = st.session_state["workspace_page"]
page_title, page_description = PAGES[page]

brand_column, workspace_column = st.columns([1.5, .55], vertical_alignment="center")
with brand_column:
    st.markdown(
        f"""
    <div class="ci-brand">
      <div class="ci-brand-left">
        <div class="ci-mark">
      <img src="data:image/png;base64,{RETAINLY_SYMBOL_DATA}" alt="Retainly connected customer-retention symbol">
        </div>
        <div>
          <div class="ci-product">Customer Intelligence &amp; Retention Desk</div>
          <div class="ci-product-subtitle">Customer care · retention operations · insights</div>
        </div>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
with workspace_column:
    st.markdown('<div class="ci-header-workspace-label">Workspace</div>', unsafe_allow_html=True)
    st.selectbox(
        "Header workspace",
        ("Real workspace", "Demo workspace"),
        key="header_workspace_mode",
        label_visibility="collapsed",
        on_change=_apply_header_workspace_mode,
    )
    st.markdown(
        f'<div class="ci-storage"><span class="ci-storage-dot"></span>'
        f'{"Demo data · synthetic" if is_demo else ("Supabase · persistent" if remote_storage_configured() else "Local storage · persistent")}</div>',
        unsafe_allow_html=True,
    )
st.markdown('<div class="ci-header-divider"></div>', unsafe_allow_html=True)
if is_demo:
    st.markdown(
        '<div class="ci-demo-banner" role="status">DEMO DATA — NOT REAL CUSTOMERS · '
        'Synthetic examples only. They are separate from uploaded files and the real case database.</div>',
        unsafe_allow_html=True,
    )
if page != "Welcome":
    st.markdown(
        f'<div class="ci-eyebrow">Customer intelligence</div><h1 class="ci-page-title">{page_title}</h1>'
        f'<div class="ci-page-copy">{page_description}</div>',
        unsafe_allow_html=True,
    )

source_df = None
source_error = None
dataset_key = None
mapping = {}
if is_demo:
    source_df = demo_customers
    dataset_key = "demo-synthetic-v1"
    mapping = detect_columns(source_df)
elif uploaded_file is not None:
    try:
        raw_df = load_data(uploaded_file)
        source_df = clean_data(raw_df)
        mapping = detect_columns(source_df)
        dataset_key = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
    except ValueError as error:
        source_error = str(error)


def _optional_date(value):
    if value is None or pd.isna(value) or value == "":
        return None
    return date.fromisoformat(str(value)[:10])


def _as_text(value):
    return "" if value is None or pd.isna(value) else str(value)


def _style_chart(figure):
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font={"family": '"Source Sans", "Segoe UI", Arial, sans-serif', "color": "#24364B", "size": 12},
        title={"font": {"size": 15, "color": "#172A43"}, "x": 0.02, "xanchor": "left"},
        margin={"l": 10, "r": 14, "t": 48, "b": 14},
        colorway=["#287FB2", "#4C9ED0", "#83C2E8", "#B8DDF3", "#9EDFD8", "#BEDDCB", "#F3E7A1", "#F4CFB5"],
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0, "font": {"size": 11}},
        hoverlabel={"bgcolor": "#FFFFFF", "bordercolor": "#C9D9E3", "font": {"color": "#172A43", "size": 12}},
    )
    figure.update_xaxes(showgrid=False, linecolor="#DDE5EA", zerolinecolor="#DDE5EA", tickfont={"color": "#536579"})
    figure.update_yaxes(gridcolor="#E9EEF2", zerolinecolor="#DDE5EA", tickfont={"color": "#536579"})
    return figure


def _filtered_cases(frame, key_prefix):
    if frame.empty:
        st.info("No cancellation review records yet. Submit a case to start building your history.")
        return frame

    dates = pd.to_datetime(frame["request_date"], errors="coerce").dropna()
    if dates.empty:
        start = end = date.today()
    else:
        start, end = dates.min().date(), dates.max().date()

    with st.container(border=True):
        filter_heading, reset_column = st.columns([4, 1], vertical_alignment="center")
        with filter_heading:
            st.markdown(
                '<div class="ci-card-title">Filter cases</div>'
                '<div class="ci-card-copy">Narrow the view using request details or current case status.</div>',
                unsafe_allow_html=True,
            )
        with reset_column:
            st.button(
                "Reset filters",
                key=f"{key_prefix}_reset_filters",
                on_click=_reset_case_filters,
                args=(key_prefix,),
                help="Clear the search and restore all filters to their default values.",
                width="stretch",
            )
        date_column, search_column = st.columns([1, 1])
        with date_column:
            date_range = st.date_input(
                "Request date range",
                value=(start, end),
                min_value=start,
                max_value=end,
                key=f"{key_prefix}_date",
            )
        with search_column:
            search = st.text_input(
                "Search case ID, customer ID, or notes",
                key=f"{key_prefix}_search",
                icon=":material/search:",
            ).strip().lower()
        reason_options = ["All"] + sorted(frame["primary_reason"].dropna().astype(str).unique().tolist())
        outcome_options = ["All"] + sorted(frame["outcome"].dropna().astype(str).unique().tolist())
        cancel_options = ["All"] + sorted(frame["cancellation_status"].dropna().astype(str).unique().tolist())
        plan_options = ["All"] + sorted(frame["plan"].dropna().astype(str).unique().tolist())
        owner_options = ["All"] + sorted(frame["owner"].dropna().astype(str).unique().tolist())
        col1, col2, col3 = st.columns(3)
        reason = col1.selectbox("Reason", reason_options, key=f"{key_prefix}_reason")
        outcome = col2.selectbox("Outcome", outcome_options, key=f"{key_prefix}_outcome")
        cancellation_status = col3.selectbox("Cancellation status", cancel_options, key=f"{key_prefix}_cancel")
        col1, col2 = st.columns(2)
        plan = col1.selectbox("Plan", plan_options, key=f"{key_prefix}_plan")
        owner = col2.selectbox("Owner", owner_options, key=f"{key_prefix}_owner")

    result = frame.copy()
    if isinstance(date_range, tuple) and len(date_range) == 2:
        request_dates = pd.to_datetime(result["request_date"], errors="coerce").dt.date
        result = result.loc[request_dates.between(date_range[0], date_range[1])]
    filters = [
        ("primary_reason", reason),
        ("outcome", outcome),
        ("cancellation_status", cancellation_status),
        ("plan", plan),
        ("owner", owner),
    ]
    for column, selected in filters:
        if selected != "All":
            result = result.loc[result[column].fillna("").astype(str) == selected]
    if search:
        search_columns = ["customer_id", "notes", "customer_comments", "case_id"]
        mask = pd.Series(False, index=result.index)
        for column in search_columns:
            mask |= result[column].fillna("").astype(str).str.lower().str.contains(search, regex=False)
        result = result.loc[mask]
    return result


def _case_summary_metrics(frame):
    request_count = len(frame)
    completed = int((frame["cancellation_status"] == "Cancelled").sum())
    saved = int((frame["outcome"] == "Saved").sum())
    pending = int((frame["outcome"] == "Pending").sum())
    resolved_count = int(frame["outcome"].isin(["Saved", "Still leaving"]).sum())
    save_rate = (saved / resolved_count * 100) if resolved_count else None
    monthly_charge = pd.to_numeric(frame["monthly_charge"], errors="coerce")
    completed_revenue = monthly_charge.loc[frame["cancellation_status"] == "Cancelled"].sum(min_count=1)
    pending_revenue = monthly_charge.loc[frame["outcome"] == "Pending"].sum(min_count=1)
    return request_count, completed, saved, pending, save_rate, completed_revenue, pending_revenue


def _outcome_group_stats(frame, group_column):
    rows = []
    for label, group in frame.groupby(group_column, dropna=False):
        requests = len(group)
        saved = int((group["outcome"] == "Saved").sum())
        resolved = int(group["outcome"].isin(["Saved", "Still leaving"]).sum())
        completed = int((group["cancellation_status"] == "Cancelled").sum())
        rows.append(
            {
                group_column.replace("_", " ").title(): "(Not recorded)" if pd.isna(label) else label,
                "Sample size (requests)": requests,
                "Completed cancellations": completed,
                "Cancellation completion rate": f"{completed / requests:.1%}" if requests else "—",
                "Resolved cases": resolved,
                "Save rate among resolved": f"{saved / resolved:.1%}" if resolved else "—",
            }
        )
    return pd.DataFrame(rows)


def _render_welcome():
    with st.container(border=True):
        hero_text, hero_art = st.columns([1.35, 0.85], vertical_alignment="center")
        with hero_text:
            st.markdown(
                '<div class="ci-eyebrow">A connected customer retention workspace</div>'
                '<h1 class="ci-welcome-title">Turn insights into lasting customer relationships</h1>'
                '<p class="ci-welcome-lede">Bring cancellation feedback, thoughtful follow-up, and customer patterns '
                'together to make every retention conversation more informed.</p>',
                unsafe_allow_html=True,
            )
            st.button(
                "Explore Retention Overview",
                key="welcome_open_overview",
                type="primary",
                on_click=_select_page,
                args=("Overview",),
            )
        with hero_art:
            st.markdown(
                """
                <div class="ci-welcome-art" role="img" aria-label="A customer connected to helpful retention actions and customer insights">
                  <svg viewBox="0 0 400 230" role="presentation" xmlns="http://www.w3.org/2000/svg">
                    <path d="M109 116c35 0 41-45 82-45M109 116h88m-88 0c35 0 41 45 82 45"
                      fill="none" stroke="#8FC7E6" stroke-width="3" stroke-linecap="round"/>
                    <circle cx="194" cy="71" r="4" fill="#F2DCE8" stroke="#234B60" stroke-width="1.5"/>
                    <circle cx="197" cy="116" r="4" fill="#A9CFF0" stroke="#234B60" stroke-width="1.5"/>
                    <circle cx="194" cy="161" r="4" fill="#F7E5D7" stroke="#234B60" stroke-width="1.5"/>
                    <circle cx="73" cy="116" r="43" fill="#E8F4FA"/>
                    <circle cx="73" cy="99" r="12" fill="#A9CFF0"/>
                    <path d="M49 138c3-17 13-26 24-26s21 9 24 26" fill="#A9CFF0"/>
                    <path d="M49 138c8 9 16 13 24 13s16-4 24-13" fill="none" stroke="#234B60" stroke-width="2" stroke-linecap="round"/>
                    <rect x="224" y="39" width="148" height="63" rx="15" fill="#FFFFFF" stroke="#D9E3E7" stroke-width="2"/>
                    <circle cx="246" cy="61" r="10" fill="#F2DCE8"/>
                    <path d="m242 61 3 3 6-7" fill="none" stroke="#234B60" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M264 57h88M264 67h58" stroke="#AAB7BD" stroke-width="4" stroke-linecap="round"/>
                    <rect x="207" y="91" width="164" height="66" rx="16" fill="#EAF6FD" stroke="#C5DFEC" stroke-width="2"/>
                    <circle cx="231" cy="113" r="10" fill="#A9CFF0"/>
                    <path d="M227 113h8m-4-4v8" stroke="#234B60" stroke-width="1.8" stroke-linecap="round"/>
                    <path d="M249 109h99M249 119h68" stroke="#9AAAB1" stroke-width="4" stroke-linecap="round"/>
                    <path d="M249 142h7v-8h8v8h8v-14h8v14h8v-6h8" fill="none" stroke="#5E9EC2" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
                    <rect x="224" y="154" width="148" height="42" rx="13" fill="#FFFFFF" stroke="#D9E3E7" stroke-width="2"/>
                    <circle cx="246" cy="175" r="9" fill="#F7E5D7"/>
                    <path d="M242 175h8m-4-4v8" stroke="#234B60" stroke-width="1.8" stroke-linecap="round"/>
                    <path d="M264 171h84M264 180h53" stroke="#AAB7BD" stroke-width="4" stroke-linecap="round"/>
                    <circle cx="119" cy="53" r="4" fill="#F2DCE8"/>
                    <circle cx="129" cy="181" r="5" fill="#A9CFF0"/>
                  </svg>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("#### How it works")
    steps = [
        (
            "Record the request",
            "Capture the cancellation request and the customer’s stated reason.",
        ),
        (
            "Plan a follow-up",
            "Select a retention action and assign an owner or follow-up date.",
        ),
        (
            "Update the outcome",
            "Record the case decision and confirm whether cancellation occurred.",
        ),
        (
            "Review patterns",
            "Explore case trends and analyze separate historical customer data.",
        ),
    ]
    step_columns = st.columns(4)
    for index, (title, description) in enumerate(steps, start=1):
        with step_columns[index - 1].container(border=True):
            st.markdown(
                f'<div class="ci-step-number">{index}</div>'
                f'<div class="ci-step-title">{title}</div>'
                f'<div class="ci-step-copy">{description}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("#### Important information")
    information = [
        (
            "Requests and churn are different",
            "A cancellation request is not confirmed churn until the actual cancellation is marked as confirmed.",
        ),
        (
            "Your data stays separated",
            "Historical CSV uploads are used separately from the cancellation case register.",
        ),
        (
            "Demo data is synthetic",
            "When the Demo workspace is selected, generated examples are not real customer information.",
        ),
        (
            "Model scores are estimates",
            "Historical model performance is descriptive and does not guarantee future results.",
        ),
    ]
    info_columns = st.columns(2)
    for index, (title, detail) in enumerate(information):
        with info_columns[index % 2].container(border=True):
            st.markdown(
                f'<div class="ci-info-card"><strong>{title}</strong><br>{detail}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("#### Go to a workspace")
    cards = [
        (
            "Retention Overview",
            "See request volume, outcomes, confirmed cancellations, and revenue exposure.",
            "Overview",
        ),
        (
            "New Cancellation Review",
            "Record a customer-stated reason and coordinate the next retention step.",
            "New Cancellation Review",
        ),
        (
            "Case History",
            "Filter, inspect, update, and export cancellation review cases.",
            "Case History",
        ),
        (
            "Churn Analysis",
            "Analyze historical labels, evaluate a model, and score eligible customers.",
            "Churn Analysis",
        ),
        (
            "Data Quality / Settings",
            "Review uploaded data quality, confirm target mapping, and read storage guidance.",
            "Data Quality/Settings",
        ),
    ]
    for start in range(0, len(cards), 2):
        card_columns = st.columns(2)
        for column, (title, description, page_name) in zip(
            card_columns, cards[start : start + 2]
        ):
            page_slug = page_name.lower().replace(" ", "_").replace("/", "_")
            with column.container(
                border=True,
                key=f"workspace_card_{page_slug}",
            ):
                st.markdown(
                    f'<div class="ci-welcome-card">'
                    f'<div class="ci-card-title">{title}</div>'
                    f'<div class="ci-card-copy">{description}</div></div>',
                    unsafe_allow_html=True,
                )
                st.button(
                    f"Open {title}",
                    key=f"welcome_card_{page_name}",
                    type="secondary",
                    width="stretch",
                    on_click=_select_page,
                    args=(page_name,),
                )


def _render_overview():
    with st.container(border=True):
        hero_text, hero_art = st.columns([1.45, .75], vertical_alignment="center")
        with hero_text:
            st.markdown(
                '<div class="ci-eyebrow">Retention workspace</div>'
                '<div class="ci-dashboard-title">Understand requests. Improve follow-up.</div>'
                '<div class="ci-dashboard-copy">See customer-stated reasons, track the next action, and distinguish '
                'cancellation requests from confirmed cancellations.</div>',
                unsafe_allow_html=True,
            )
            st.button(
                "Create cancellation review",
                key="overview_hero_new_case",
                type="primary",
                on_click=_select_page,
                args=("New Cancellation Review",),
            )
        with hero_art:
            st.markdown(
                """
                <div class="ci-dashboard-art" role="img" aria-label="Customer requests connected to follow-up and outcome insights">
                  <svg viewBox="0 0 320 150" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                    <path d="M66 76h72m0 0 48-37m-48 37 48 37" fill="none" stroke="#75B9DA" stroke-width="3" stroke-linecap="round"/>
                    <circle cx="66" cy="76" r="26" fill="#D9EFFA"/>
                    <circle cx="66" cy="68" r="7" fill="#4C9ED0"/>
                    <path d="M53 91c2-9 7-14 13-14s11 5 13 14" fill="#4C9ED0"/>
                    <rect x="182" y="16" width="112" height="43" rx="12" fill="#FFFFFF" stroke="#C9E2F0" stroke-width="2"/>
                    <circle cx="198" cy="37" r="7" fill="#F3E7A1"/>
                    <path d="m195 37 2 2 4-5m10-1h65m-65 8h42" fill="none" stroke="#416A83" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    <rect x="182" y="91" width="112" height="43" rx="12" fill="#FFFFFF" stroke="#C9E2F0" stroke-width="2"/>
                    <circle cx="198" cy="112" r="7" fill="#CDE9DF"/>
                    <path d="M211 120v-9h9v9m3 0V99h9v21m3 0v-14h9v14m3 0v-7h9v7" fill="none" stroke="#287FB2" stroke-width="2" stroke-linejoin="round"/>
                  </svg>
                </div>
                """,
                unsafe_allow_html=True,
            )

    cases = demo_cases.copy() if is_demo else list_cases()
    if cases.empty:
        with st.container(border=True):
            st.markdown('<div class="ci-card-title">Your retention workspace is ready</div>', unsafe_allow_html=True)
            st.write("No cancellation reviews have been recorded yet. Add a first case to start building an operational view of customer requests and outcomes.")
            st.button(
                "Create a cancellation review",
                key="overview_create_case",
                type="primary",
                on_click=_select_page,
                args=("New Cancellation Review",),
            )
        with st.container(border=True):
            st.markdown('<div class="ci-status-note">A cancellation request is not counted as churn. Churn is recorded only when actual cancellation is confirmed.</div>', unsafe_allow_html=True)
        return

    filtered = _filtered_cases(cases, "overview")
    if filtered.empty:
        with st.container(border=True):
            st.info("No cases match these filters. Adjust the date range or filters to continue.")
        return
    requests, completed, saved, pending, save_rate, completed_revenue, pending_revenue = _case_summary_metrics(filtered)
    st.markdown("#### Case outcomes")
    cols = st.columns(6)
    mrr_exposure = pd.to_numeric(
        filtered.loc[
            filtered["cancellation_status"].eq("Cancelled")
            | filtered["outcome"].eq("Pending"),
            "monthly_charge",
        ],
        errors="coerce",
    ).sum(min_count=1)
    metric_data = [
        ("Cancellation requests", f"{requests:,}", "Requests recorded in the selected date range. A request alone is not churn."),
        ("Confirmed cancellations", f"{completed:,}", "Cases explicitly marked as actually cancelled."),
        ("Saved cases", f"{saved:,}", "Cases whose current outcome is Saved, irrespective of cancellation-status history."),
        ("Pending follow-ups", f"{pending:,}", "Cases still awaiting a decided outcome."),
        ("Save rate", f"{save_rate:.1f}%" if save_rate is not None else "N/A", "Saved / (Saved + Still leaving). Pending cases are excluded."),
        ("MRR exposure", f"{mrr_exposure:,.0f}" if pd.notna(mrr_exposure) else "N/A", "Recorded monthly charges associated with confirmed cancellations or pending cases; illustrative units, not lifetime value."),
    ]
    metric_icons = [
        '<path d="M7 3.75h7l4 4v12.5H7z"/><path d="M14 3.75v4h4M10 12h5m-5 3.5h5"/>',
        '<path d="M12 20s-7-4.2-7-9a4 4 0 0 1 7-2.6A4 4 0 0 1 19 11c0 4.8-7 9-7 9Z"/><path d="m9.5 12 1.7 1.7 3.6-3.8"/>',
        '<path d="M5 5.5h14v14H5z"/><path d="M8 3.75v3.5m8-3.5v3.5M5 9.5h14"/><path d="m9 14 2 2 4-4"/>',
        '<path d="M5 5.5h14v14H5z"/><path d="M8 3.75v3.5m8-3.5v3.5M5 9.5h14m3 2.5v4"/>',
        '<path d="M5 18.5h14"/><path d="m7 15 3-3 2.5 2 4.5-6"/><path d="M14 8h3v3"/>',
        '<path d="M12 4v16m4-12.5c-.7-1-2-1.5-4-1.5-2.2 0-3.5 1-3.5 2.5S10 11 12 11s3.5 1 3.5 2.5S14.2 16 12 16c-2 0-3.3-.5-4-1.5"/>',
    ]
    for column, (label, value, help_text), icon_path in zip(cols, metric_data, metric_icons):
        column.markdown(
            f'<section class="ci-kpi-card" role="group" aria-label="{label}: {value}" title="{help_text}">'
            f'<div class="ci-kpi-heading"><span class="ci-kpi-icon" aria-hidden="true">'
            f'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" '
            f'stroke-linecap="round" stroke-linejoin="round">{icon_path}</svg></span>'
            f'<span class="ci-kpi-label">{label}</span></div>'
            f'<div class="ci-kpi-value">{value}</div></section>',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div class="ci-status-note">Requests, case outcomes, and actual cancellation status are distinct. '
        'Completion rate means confirmed cancellations divided by requests; group comparisons are descriptive and do not establish causation.</div>',
        unsafe_allow_html=True,
    )

    reason_counts = filtered.groupby("primary_reason").size().reset_index(name="Requests")
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown('<div class="ci-card-title">Request volume by reason</div><div class="ci-card-copy">Customer-stated primary reasons in the selected cases.</div>', unsafe_allow_html=True)
            fig = px.bar(
                reason_counts.sort_values("Requests"),
                x="Requests",
                y="primary_reason",
                orientation="h",
                labels={"primary_reason": "Primary reason", "Requests": "Requests"},
                text="Requests",
                color_discrete_sequence=["#4C9ED0"],
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(_style_chart(fig), width="stretch")
    with right:
        with st.container(border=True):
            st.markdown('<div class="ci-card-title">Cancellation request trend</div><div class="ci-card-copy">Requests and confirmed cancellations by request month.</div>', unsafe_allow_html=True)
            trend_data = filtered.copy()
            trend_data["request_month"] = pd.to_datetime(trend_data["request_date"], errors="coerce").dt.to_period("M").astype(str)
            trend = trend_data.dropna(subset=["request_month"]).groupby("request_month").agg(
                requests=("case_id", "count"),
                confirmed_cancellations=("cancellation_status", lambda values: int((values == "Cancelled").sum())),
            ).reset_index()
            if trend.empty:
                st.info("No dated requests are available for this trend.")
            else:
                fig = px.line(
                    trend,
                    x="request_month",
                    y=["requests", "confirmed_cancellations"],
                    markers=True,
                    labels={"request_month": "Request month", "value": "Cases", "variable": "Measure"},
                    color_discrete_map={"requests": "#287FB2", "confirmed_cancellations": "#83C2E8"},
                )
                st.plotly_chart(_style_chart(fig), width="stretch")

    reason_trend = filtered.copy()
    reason_trend["request_month"] = pd.to_datetime(reason_trend["request_date"], errors="coerce").dt.to_period("M").astype(str)
    reason_trend = reason_trend.dropna(subset=["request_month"]).groupby(
        ["request_month", "primary_reason"]
    ).size().reset_index(name="Requests")
    if not reason_trend.empty:
        with st.container(border=True):
            st.markdown('<div class="ci-card-title">Reasons over time</div><div class="ci-card-copy">Monthly request volume by customer-stated reason.</div>', unsafe_allow_html=True)
            fig = px.bar(
                reason_trend,
                x="request_month",
                y="Requests",
                color="primary_reason",
                labels={"request_month": "Request month", "primary_reason": "Primary reason"},
                color_discrete_sequence=[
                    "#287FB2",
                    "#4C9ED0",
                    "#83C2E8",
                    "#B8DDF3",
                    "#9EDFD8",
                    "#BEDDCB",
                    "#E5CD65",
                    "#F4CFB5",
                ],
            )
            st.plotly_chart(_style_chart(fig), width="stretch")

    st.markdown("#### Outcome and retention patterns")
    outcome_cols = st.columns(2)
    with outcome_cols[0]:
        with st.container(border=True):
            st.markdown('<div class="ci-card-title">Case outcomes by reason</div><div class="ci-card-copy">Counts by current outcome; requests are not equivalent to completed cancellations.</div>', unsafe_allow_html=True)
            by_reason = filtered.groupby(["primary_reason", "outcome"]).size().reset_index(name="Cases")
            outcome_chart = px.bar(
                by_reason,
                x="primary_reason",
                y="Cases",
                color="outcome",
                barmode="group",
                text_auto=True,
                labels={"primary_reason": "Primary reason", "outcome": "Case outcome"},
                color_discrete_map={"Pending": "#E5CD65", "Saved": "#75BCA9", "Still leaving": "#4C9ED0"},
            )
            st.plotly_chart(_style_chart(outcome_chart), width="stretch")
            st.caption("Sample sizes are shown in the segment table below.")
    with outcome_cols[1]:
        with st.container(border=True):
            st.markdown('<div class="ci-card-title">Save rate by retention action</div><div class="ci-card-copy">Saved cases divided by decided cases for each action.</div>', unsafe_allow_html=True)
            decided = filtered.loc[filtered["outcome"].isin(["Saved", "Still leaving"])].copy()
            if decided.empty:
                st.info("Save-rate comparisons appear after at least one case has a decided outcome.")
            else:
                action_stats = decided.groupby("retention_action").agg(
                    decided_cases=("case_id", "count"),
                    saved_cases=("outcome", lambda values: int((values == "Saved").sum())),
                ).reset_index()
                action_stats["save_rate"] = action_stats["saved_cases"] / action_stats["decided_cases"]
                action_stats["sample"] = action_stats["decided_cases"].map(lambda value: f"n={value}")
                action_chart = px.bar(
                    action_stats,
                    x="retention_action",
                    y="save_rate",
                    text="sample",
                    hover_data={"saved_cases": True, "decided_cases": True, "save_rate": ":.1%"},
                    labels={"retention_action": "Retention action", "save_rate": "Save rate"},
                    color_discrete_sequence=["#4C9ED0"],
                )
                action_chart.update_yaxes(tickformat=".0%", range=[0, 1])
                st.plotly_chart(_style_chart(action_chart), width="stretch")
                st.caption("Descriptive comparison only; sample size and selection effects can influence results.")

    reason_counts_by_name = filtered["primary_reason"].value_counts()
    top_reason = reason_counts_by_name.index[0] if not reason_counts_by_name.empty else "No recorded reason"
    top_reason_count = int(reason_counts_by_name.iloc[0]) if not reason_counts_by_name.empty else 0
    top_reason_share = top_reason_count / requests if requests else 0
    confirmed_share = completed / requests if requests else 0
    resolved_cases = int(filtered["outcome"].isin(["Saved", "Still leaving"]).sum())
    insights = [
        (
            "Most common stated reason",
            f"{top_reason}: {top_reason_count:,} of {requests:,} requests ({top_reason_share:.1%}).",
        ),
        (
            "Confirmed cancellations",
            f"{completed:,} of {requests:,} requests are marked as actually cancelled ({confirmed_share:.1%}).",
        ),
        (
            "Follow-up still pending",
            f"{pending:,} case(s) have a Pending outcome; these are not counted as confirmed churn.",
        ),
    ]
    if save_rate is not None:
        insights.append(
            ("Save rate among decided cases", f"{save_rate:.1f}% across {resolved_cases:,} decided case(s).")
        )
    if is_demo:
        demo_churn_rate = float(demo_customers["churn_target"].mean())
        insights.append(
            ("Illustrative demo churn-label share", f"{demo_churn_rate:.1%} of synthetic customer profiles; not a real-world forecast.")
        )
    st.markdown("#### What stands out")
    st.caption(
        "Descriptive observations from the currently filtered records; these patterns do not establish causes."
    )
    for start in range(0, len(insights), 3):
        insight_row = st.columns(min(3, len(insights) - start))
        for column, (title, description) in zip(insight_row, insights[start : start + 3]):
            with column:
                st.markdown(
                    f'<div class="ci-insight-card"><div class="ci-insight-title">{escape(title)}</div>'
                    f'<div class="ci-insight-copy">{escape(description)}</div></div>',
                    unsafe_allow_html=True,
                )
    st.button(
        "Explore cases in Case History",
        key="overview_insights_case_history",
        type="secondary",
        on_click=_select_page,
        args=("Case History",),
    )
    if is_demo:
        with st.expander("About demo data"):
            st.write(
                "All displayed customer rows, cancellation cases, labels, and metrics are generated synthetic examples. "
                "The fixed random seed makes the customer feature and churn sample reproducible. "
                "The churn outcome is sampled with noise from tenure, contract, satisfaction, engagement, support contacts, and escalations. "
                "These examples are not real customer records, research evidence, or predictions of real-world behavior."
            )

    st.markdown("#### Monthly revenue exposure")
    revenue_cols = st.columns(2)
    with revenue_cols[0].container(border=True):
        st.metric(
            "Confirmed monthly revenue associated with cancellations",
            f"{completed_revenue:,.2f} / month" if pd.notna(completed_revenue) else "Not available",
            help="Sum of recorded monthly charges for cases marked actually cancelled. In the currency/units entered; not lifetime value.",
        )
        st.caption("Confirmed cancellation status only · excludes cases without a monthly charge.")
    with revenue_cols[1].container(border=True):
        st.metric(
            "Pending monthly revenue estimate",
            f"{pending_revenue:,.2f} / month" if pd.notna(pending_revenue) else "Not available",
            help="Sum of recorded charges for cases with a Pending outcome; not confirmed churn or revenue loss.",
        )
        st.caption("Estimated exposure only · pending cases have not been treated as churn.")

    st.markdown("#### Outcome rates by segment")
    stat_columns = st.columns(3)
    for column, (segment, label) in zip(
        stat_columns,
        [("primary_reason", "Reason"), ("plan", "Plan"), ("retention_action", "Retention action")],
    ):
        with column:
            with st.container(border=True):
                st.markdown(f'<div class="ci-card-title">By {label.lower()}</div><div class="ci-card-copy">Counts and rates include sample sizes.</div>', unsafe_allow_html=True)
                segment_data = filtered.copy()
                segment_data[segment] = segment_data[segment].fillna("(Not recorded)")
                stats = _outcome_group_stats(segment_data, segment)
                st.dataframe(stats, hide_index=True, width="stretch")

    if is_demo:
        with st.container(border=True):
            st.markdown(
                '<div class="ci-card-title">Recent demo cancellation reviews</div>'
                '<div class="ci-card-copy">Synthetic examples only · read-only in the demo workspace.</div>',
                unsafe_allow_html=True,
            )
            preview = filtered.sort_values(
                ["request_date", "case_id"], ascending=[False, False]
            ).head(8)
            st.dataframe(
                preview[
                    [
                        "customer_id",
                        "request_date",
                        "primary_reason",
                        "retention_action",
                        "outcome",
                        "cancellation_status",
                        "data_classification",
                    ]
                ],
                hide_index=True,
                width="stretch",
            )


def _render_new_case():
    if is_demo:
        st.info(
            "DEMO DATA — NOT REAL CUSTOMERS. Demo reviews are generated read-only examples. "
            "Switch to the Real workspace to create a persistent cancellation review."
        )
        return
    st.markdown(
        '<div class="ci-status-note">Record the request and customer-stated reason first. A case outcome and actual cancellation status are recorded separately.</div>',
        unsafe_allow_html=True,
    )
    if st.session_state.pop("_reset_duplicate_confirmation", False):
        st.session_state["allow_duplicate_case"] = False
    with st.form("new_cancellation_case"):
        st.markdown('<div class="ci-eyebrow">01 · Customer and subscription</div>', unsafe_allow_html=True)
        st.caption("Required fields are marked with *. Enter tenure and monthly charge only when known.")
        col1, col2, col3 = st.columns(3)
        customer_id = col1.text_input("Customer ID *", max_chars=200)
        request_date = col2.date_input("Request date *", value=date.today(), max_value=date.today())
        plan = col3.text_input("Plan or subscription")
        contract_type = col1.text_input("Contract type")
        tenure_known = col2.checkbox("Tenure available")
        tenure = col2.number_input("Tenure (months)", min_value=0.0, value=0.0, disabled=not tenure_known)
        charge_known = col3.checkbox("Monthly charge available")
        monthly_charge = col3.number_input("Monthly charge (currency as recorded)", min_value=0.0, value=0.0, disabled=not charge_known)

        st.markdown('<div class="ci-eyebrow">02 · Customer-stated reason</div>', unsafe_allow_html=True)
        st.caption("Keep the primary reason faithful to the customer's explanation; use comments for additional context.")
        col1, col2 = st.columns(2)
        primary_reason = col1.selectbox("Primary reason *", [REASON_PLACEHOLDER] + REASONS)
        secondary_reason = col2.selectbox("Secondary reason", ["None"] + REASONS)
        customer_comments = st.text_area("Customer comments (optional)", max_chars=5000)
        stay_request = st.text_area("What could make you stay? (optional)", max_chars=3000)
        st.markdown('<div class="ci-eyebrow">03 · Retention and status</div>', unsafe_allow_html=True)
        st.caption("Retention action is what was offered. Outcome is the current case decision. Cancellation status records whether the customer actually left.")
        col1, col2, col3 = st.columns(3)
        retention_action = col1.selectbox(
            "Retention action offered",
            RETENTION_ACTIONS,
            index=RETENTION_ACTIONS.index("None"),
        )
        outcome = col2.selectbox("Case outcome", OUTCOMES)
        cancellation_status = col3.selectbox("Actual cancellation status", CANCELLATION_STATUSES)
        cancellation_date_input = st.date_input(
            "Actual cancellation date (only when status is Cancelled)",
            value=date.today(),
            min_value=request_date,
            max_value=date.today(),
        )
        cancellation_date = cancellation_date_input if cancellation_status == "Cancelled" else None
        col1, col2 = st.columns(2)
        owner = col1.text_input("Owner (optional)")
        follow_up_enabled = col2.checkbox("Set a follow-up date")
        follow_up_date = None
        if follow_up_enabled:
            follow_up_date = col2.date_input("Follow-up date", value=max(request_date, date.today()), min_value=request_date)
        allow_duplicate = st.checkbox(
            "I confirm this is a separate request if a matching case already exists",
            key="allow_duplicate_case",
        )
        submitted = st.form_submit_button("Save cancellation case", type="primary")

    if not submitted:
        return
    if not customer_id.strip():
        st.error("Enter a customer ID before saving.")
        return
    if primary_reason == REASON_PLACEHOLDER:
        st.error("Select the customer's primary cancellation reason before saving.")
        return
    if cancellation_status == "Cancelled" and cancellation_date is None:
        st.error("A cancellation date is required when actual status is Cancelled.")
        return
    duplicate_id = find_duplicate(customer_id.strip(), request_date.isoformat(), primary_reason)
    if duplicate_id is not None and not allow_duplicate:
        st.warning(
            f"A case with this customer ID, request date, and primary reason already exists (case #{duplicate_id}). "
            "Enable the separate-request confirmation and submit again only if this is intentional."
        )
        return
    record = {
        "customer_id": customer_id.strip(),
        "request_date": request_date.isoformat(),
        "plan": plan.strip() or None,
        "contract_type": contract_type.strip() or None,
        "tenure": tenure if tenure_known else None,
        "monthly_charge": monthly_charge if charge_known else None,
        "primary_reason": primary_reason,
        "secondary_reason": None if secondary_reason == "None" else secondary_reason,
        "customer_comments": customer_comments.strip() or None,
        "stay_request": stay_request.strip() or None,
        "retention_action": retention_action,
        "outcome": outcome,
        "cancellation_status": cancellation_status,
        "cancellation_date": cancellation_date.isoformat() if cancellation_date else None,
        "owner": owner.strip() or None,
        "follow_up_date": follow_up_date.isoformat() if follow_up_date else None,
    }
    try:
        case_id = create_case(record)
    except sqlite3.Error:
        st.error("The case could not be saved. Check database access and try again.")
        return
    st.session_state["_reset_duplicate_confirmation"] = True
    st.success(f"Cancellation review saved successfully. Case ID: **#{case_id}**")


def _render_case_history():
    cases = demo_cases.copy() if is_demo else list_cases()
    if cases.empty:
        with st.container(border=True):
            st.markdown('<div class="ci-card-title">No cancellation cases yet</div>', unsafe_allow_html=True)
            st.write("Record a customer’s request to begin tracking follow-ups, outcomes, and confirmed cancellations.")
            st.button(
                "Create a cancellation review",
                key="history_create_case",
                type="primary",
                on_click=_select_page,
                args=("New Cancellation Review",),
            )
        return
    filtered = _filtered_cases(cases, "history")
    st.caption(f"{len(filtered):,} matching case(s)")
    if filtered.empty:
        st.info("No cases match the selected filters.")
        return
    columns = [
        "case_id", "customer_id", "request_date", "plan", "primary_reason",
        "retention_action", "outcome", "cancellation_status", "cancellation_date",
        "owner", "follow_up_date", "updated_at",
    ]
    display_cases = filtered[columns].copy()
    display_cases = display_cases.rename(
        columns={
            "case_id": "Case ID",
            "customer_id": "Customer ID",
            "request_date": "Request date",
            "plan": "Plan",
            "primary_reason": "Primary reason",
            "retention_action": "Retention action",
            "outcome": "Outcome",
            "cancellation_status": "Actual cancellation",
            "cancellation_date": "Cancellation date",
            "owner": "Owner",
            "follow_up_date": "Follow-up date",
            "updated_at": "Last updated",
        }
    )
    for column in ["Request date", "Cancellation date", "Follow-up date"]:
        display_cases[column] = pd.to_datetime(display_cases[column], errors="coerce")
    display_cases["Last updated"] = pd.to_datetime(display_cases["Last updated"], errors="coerce", utc=True)
    with st.container(border=True):
        st.markdown('<div class="ci-card-title">Cancellation cases</div><div class="ci-card-copy">Original reasons remain unchanged as outcomes and follow-ups are updated.</div>', unsafe_allow_html=True)
        st.dataframe(
            display_cases,
            hide_index=True,
            width="stretch",
            column_config={
                "Case ID": st.column_config.NumberColumn("Case ID", format="%d"),
                "Request date": st.column_config.DateColumn("Request date", format="MMM D, YYYY"),
                "Cancellation date": st.column_config.DateColumn("Cancellation date", format="MMM D, YYYY"),
                "Follow-up date": st.column_config.DateColumn("Follow-up date", format="MMM D, YYYY"),
                "Last updated": st.column_config.DatetimeColumn("Last updated", format="MMM D, YYYY h:mm a"),
            },
        )
    export_cases = filtered.copy()
    file_name = "cancellation_cases_filtered.csv"
    if is_demo:
        export_cases["data_classification"] = "DEMO DATA — NOT REAL CUSTOMERS"
        file_name = "demo_cancellation_cases_DEMO_DATA_NOT_REAL_CUSTOMERS.csv"
    csv_bytes = export_cases.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Export filtered cases as CSV",
        data=csv_bytes,
        file_name=file_name,
        mime="text/csv",
    )

    if is_demo:
        st.caption(
            "DEMO DATA — NOT REAL CUSTOMERS. Demo reviews are read-only and are not stored in the production case database."
        )
        return

    st.divider()
    st.markdown("#### Inspect and update")
    customer_by_case = dict(zip(filtered["case_id"].astype(int), filtered["customer_id"].astype(str)))
    selected_id = st.selectbox(
        "Case ID",
        filtered["case_id"].astype(int).tolist(),
        format_func=lambda value: f"#{value} — customer {customer_by_case.get(value, '')}",
    )
    selected = get_case(int(selected_id))
    if selected is None:
        st.error("The selected case is no longer available. Refresh the page.")
        return
    with st.expander("Original request details · customer-stated reason is preserved", expanded=True):
        st.write(
            {
                "Customer ID": selected["customer_id"],
                "Request date": selected["request_date"],
                "Primary reason": selected["primary_reason"],
                "Secondary reason": selected["secondary_reason"],
                "Customer comments": selected["customer_comments"],
                "What could make you stay?": selected["stay_request"],
            }
        )
    st.markdown('<div class="ci-card-title">Case follow-up</div><div class="ci-card-copy">Update the retention response, outcome, cancellation status, owner, dates, and internal notes. The original stated reason is locked.</div>', unsafe_allow_html=True)
    with st.form(f"update_case_{selected_id}"):
        col1, col2, col3 = st.columns(3)
        action_index = RETENTION_ACTIONS.index(selected["retention_action"]) if selected["retention_action"] in RETENTION_ACTIONS else 0
        outcome_index = OUTCOMES.index(selected["outcome"]) if selected["outcome"] in OUTCOMES else 0
        status_index = CANCELLATION_STATUSES.index(selected["cancellation_status"]) if selected["cancellation_status"] in CANCELLATION_STATUSES else 0
        next_action = col1.selectbox("Retention action", RETENTION_ACTIONS, index=action_index)
        next_outcome = col2.selectbox("Outcome", OUTCOMES, index=outcome_index)
        next_status = col3.selectbox("Actual cancellation status", CANCELLATION_STATUSES, index=status_index)
        existing_cancel_date = _optional_date(selected["cancellation_date"])
        next_cancel_date_input = st.date_input(
            "Actual cancellation date (only when status is Cancelled)",
            value=existing_cancel_date or max(date.today(), date.fromisoformat(selected["request_date"])),
            min_value=date.fromisoformat(selected["request_date"]),
            max_value=date.today(),
        )
        next_cancel_date = next_cancel_date_input if next_status == "Cancelled" else None
        col1, col2 = st.columns(2)
        next_owner = col1.text_input("Owner", value=_as_text(selected["owner"]))
        follow_up_enabled = col2.checkbox("Set follow-up date", value=bool(selected["follow_up_date"]))
        next_follow_up = None
        if follow_up_enabled:
            min_followup = date.fromisoformat(selected["request_date"])
            next_follow_up = col2.date_input(
                "Follow-up date",
                value=_optional_date(selected["follow_up_date"]) or max(min_followup, date.today()),
                min_value=min_followup,
            )
        next_notes = st.text_area("Internal case notes", value=_as_text(selected["notes"]), max_chars=5000)
        save_updates = st.form_submit_button("Save case updates", type="primary")
    if save_updates:
        updates = {
            "retention_action": next_action,
            "outcome": next_outcome,
            "cancellation_status": next_status,
            "cancellation_date": next_cancel_date.isoformat() if next_cancel_date else None,
            "owner": next_owner.strip() or None,
            "follow_up_date": next_follow_up.isoformat() if next_follow_up else None,
            "notes": next_notes.strip() or None,
        }
        try:
            update_case(int(selected_id), updates)
        except (sqlite3.Error, ValueError):
            st.error("The case could not be updated. Check the selected values and try again.")
            return
        st.success(f"Case #{selected_id} updated. The original stated reason has not been changed.")


def _render_quality():
    if is_demo:
        info = get_basic_info(demo_customers)
        st.markdown("#### Synthetic demo data quality")
        st.info(
            "DEMO DATA — NOT REAL CUSTOMERS. This deterministic sample is generated in memory and is separate from uploaded files and SQLite."
        )
        cols = st.columns(4)
        cols[0].metric("Synthetic customers", f"{info['rows']:,}")
        cols[1].metric("Fields", f"{info['columns']:,}")
        cols[2].metric("Missing cells", f"{info['missing_values']:,}")
        cols[3].metric("Duplicate rows", f"{info['duplicates']:,}")
        quality = pd.DataFrame(
            {
                "Field": demo_customers.columns,
                "Type": [str(dtype) for dtype in demo_customers.dtypes],
                "Missing values": [
                    int(demo_customers[column].isna().sum())
                    for column in demo_customers.columns
                ],
                "Distinct values": [
                    int(demo_customers[column].nunique(dropna=True))
                    for column in demo_customers.columns
                ],
            }
        )
        with st.container(border=True):
            st.markdown(
                '<div class="ci-card-title">Synthetic field quality</div>'
                '<div class="ci-card-copy">Generated customer profile fields and illustrative churn labels.</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(quality, hide_index=True, width="stretch")
        with st.expander("About demo data · data dictionary"):
            st.write(
                "The fixed-seed generator creates 480 synthetic profiles and a separate set of synthetic cancellation reviews. "
                "Churn probability varies noisily with selected profile attributes. The generated data is illustrative only, "
                "not real customer information, research evidence, or a forecast."
            )
            st.markdown(
                "- **customer_id** — synthetic ID; excluded from model features.\n"
                "- **tenure** — illustrative months as a customer.\n"
                "- **contract_type / plan** — generated subscription grouping.\n"
                "- **monthly_charge** — illustrative monthly amount.\n"
                "- **support_contacts / escalations** — generated service-contact counts.\n"
                "- **satisfaction_score / engagement_score** — illustrative 1–5 and 0–100 scores.\n"
                "- **churn_target** — synthetic binary outcome used only in the demo model."
            )
        export_data = demo_customers.copy()
        export_data.insert(0, "data_classification", "DEMO DATA — NOT REAL CUSTOMERS")
        st.download_button(
            "Download synthetic demo customers",
            data=export_data.to_csv(index=False).encode("utf-8-sig"),
            file_name="demo_customers_DEMO_DATA_NOT_REAL_CUSTOMERS.csv",
            mime="text/csv",
        )
        return

    if source_df is None:
        with st.container(border=True):
            st.markdown('<div class="ci-card-title">Bring in historical customer data</div><div class="ci-card-copy">Use the CSV uploader in the left panel to inspect data quality and confirm a historical churn target. Imported files are not added to the cancellation case database.</div>', unsafe_allow_html=True)
            if source_error:
                st.error(source_error)
            else:
                st.info("No CSV is loaded yet. Case records continue to be stored separately in local persistent storage.")
        return

    st.markdown("#### Imported data quality")
    info = get_basic_info(source_df)
    cols = st.columns(4)
    cols[0].metric("Rows", f"{info['rows']:,}")
    cols[1].metric("Columns", f"{info['columns']:,}")
    cols[2].metric("Missing cells", f"{info['missing_values']:,}")
    cols[3].metric("Duplicate rows", f"{info['duplicates']:,}")
    quality = pd.DataFrame(
        {
            "Column": source_df.columns,
            "Inferred type": [str(dtype) for dtype in source_df.dtypes],
            "Missing values": [int(source_df[column].isna().sum()) for column in source_df.columns],
            "Missing %": [f"{source_df[column].isna().mean():.1%}" for column in source_df.columns],
            "Distinct values": [int(source_df[column].nunique(dropna=True)) for column in source_df.columns],
        }
    )
    with st.container(border=True):
        st.markdown('<div class="ci-card-title">Column-level quality</div><div class="ci-card-copy">Missing values, inferred types, and distinct-value counts. Duplicate rows are counted, not removed.</div>', unsafe_allow_html=True)
        st.dataframe(quality, hide_index=True, width="stretch")

    st.markdown("#### Column mapping")
    st.write("Detected mappings are suggestions. Confirm the historical churn target before analysis or model training.")
    if mapping:
        mapping_rows = [
            {"Data field": standard.replace("_", " ").title(), "CSV column": column}
            for standard, column in mapping.items()
        ]
        with st.container(border=True):
            st.dataframe(pd.DataFrame(mapping_rows), hide_index=True, width="stretch")
    else:
        st.info("No standard columns were detected automatically. You can still select the historical target below.")
    target_options = ["— Select a target —"] + list(source_df.columns)
    default_target = st.session_state.get("confirmed_target_column")
    if default_target not in source_df.columns:
        default_target = mapping.get("churn")
    default_index = target_options.index(default_target) if default_target in target_options else 0
    with st.form("target_mapping_form"):
        selected_target = st.selectbox("Historical churn target", target_options, index=default_index)
        confirm_mapping = st.form_submit_button("Confirm target mapping", type="primary")
    if confirm_mapping:
        if selected_target == "— Select a target —":
            st.session_state.pop("confirmed_target_column", None)
            st.session_state["target_mapping_confirmed"] = False
            st.warning("Select a target column to confirm the mapping.")
        else:
            st.session_state["confirmed_target_column"] = selected_target
            st.session_state["target_mapping_confirmed"] = True
            st.session_state["confirmed_dataset_key"] = dataset_key
            st.success(f"Confirmed `{selected_target}` as the historical churn target for this uploaded file.")

    if st.session_state.get("confirmed_dataset_key") == dataset_key and st.session_state.get("target_mapping_confirmed"):
        selected = st.session_state["confirmed_target_column"]
        _, unmapped = convert_churn_target(source_df, selected, return_unmapped=True)
        if unmapped:
            st.warning(f"Target values that could not be mapped to 0/1: {unmapped}")
        else:
            st.success("All non-empty target values map to a binary churn label.")
    st.markdown("#### Case storage and backups")
    with st.container(border=True):
        if remote_storage_configured():
            st.markdown('<div class="ci-card-title">Supabase · persistent cloud storage</div><div class="ci-card-copy">Cancellation reviews are stored in the configured private Postgres database and remain separate from uploaded historical data.</div>', unsafe_allow_html=True)
            st.write("Manage backups and access through the Supabase project. Keep the service-role key only in Streamlit secrets and invite trusted viewers only.")
        else:
            st.markdown('<div class="ci-card-title">Local storage · persistent</div><div class="ci-card-copy">Cancellation reviews are stored separately from uploaded historical data.</div>', unsafe_allow_html=True)
            st.write("Back up the SQLite database while the app is stopped. Keep backups access-controlled because case records can include customer identifiers and comments.")
            st.caption("For a custom local storage location, configure the CANCELLATION_DB_PATH environment variable before launch. The local path is intentionally not displayed in the app.")


def _render_demo_churn():
    st.info(
        "DEMO DATA — NOT REAL CUSTOMERS. Model results below use synthetic outcomes and are illustrative, not evidence about real customer behavior."
    )
    info = get_basic_info(demo_customers)
    positives = int(demo_customers["churn_target"].sum())
    metric_cols = st.columns(3)
    metric_cols[0].metric("Synthetic customer profiles", f"{info['rows']:,}")
    metric_cols[1].metric("Synthetic churn labels", f"{positives:,}")
    metric_cols[2].metric("Synthetic churn-label share", f"{positives / info['rows']:.1%}")

    segment_data = demo_customers.groupby(["contract_type", "churn_target"]).size().reset_index(name="Customers")
    segment_data["Synthetic label"] = segment_data["churn_target"].map(
        {0: "Active in demo", 1: "Churned in demo"}
    )
    with st.container(border=True):
        st.markdown(
            '<div class="ci-card-title">Synthetic labels by contract</div>'
            '<div class="ci-card-copy">Generated sample counts; descriptive only.</div>',
            unsafe_allow_html=True,
        )
        chart = px.bar(
            segment_data,
            x="contract_type",
            y="Customers",
            color="Synthetic label",
            barmode="group",
            text_auto=True,
            labels={"contract_type": "Contract", "Synthetic label": "Demo outcome"},
            color_discrete_map={"Active in demo": "#A9D8F5", "Churned in demo": "#4C9ED0"},
        )
        st.plotly_chart(_style_chart(chart), width="stretch")

    features, excluded = eligible_feature_columns(demo_customers, "churn_target")
    with st.expander("Exact demo model inputs and exclusions"):
        st.markdown("**Features used**")
        st.code(", ".join(map(str, features)), language=None)
        st.markdown("**Excluded columns and reasons**")
        st.write(excluded)
        st.caption(
            "Customer IDs and churn outcomes are excluded. Cancellation reasons and dates are not part of this generated customer dataset."
        )

    threshold = st.slider(
        "Demo decision threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
        key="demo_decision_threshold",
        help="Adjusts the illustrative classification metrics on the held-out synthetic sample.",
    )
    if st.button("Train and evaluate demo model", type="primary"):
        try:
            training_data, holdout_data = train_test_split(
                demo_customers,
                test_size=0.25,
                random_state=2026,
                stratify=demo_customers["churn_target"],
            )
            fitted_model, training_metrics = train_model(
                training_data,
                target="churn_target",
            )
            holdout_scores = score_customers(
                fitted_model,
                holdout_data,
                training_metrics["feature_columns"],
            )
        except (ValueError, TypeError, KeyError) as error:
            st.error(f"The synthetic demo model could not be evaluated: {error}")
            return
        st.session_state["demo_model"] = fitted_model
        st.session_state["demo_model_metrics"] = training_metrics
        st.session_state["demo_test_scores"] = pd.DataFrame(
            {
                "customer_id": holdout_data["customer_id"].to_numpy(),
                "actual_demo_churn_target": holdout_data["churn_target"].to_numpy(),
                "predicted_demo_churn_probability": holdout_scores,
            }
        )

    test_scores = st.session_state.get("demo_test_scores")
    training_metrics = st.session_state.get("demo_model_metrics")
    if test_scores is None or training_metrics is None:
        st.caption(
            "Train to fit on a reproducible synthetic training split, evaluate on held-out synthetic customers, and see their risk scores."
        )
        return

    holdout_metrics = evaluate_threshold(
        test_scores["actual_demo_churn_target"].to_numpy(),
        test_scores["predicted_demo_churn_probability"].to_numpy(),
        threshold,
    )
    st.markdown("#### Held-out demo evaluation · illustrative only")
    st.caption(
        f"Stratified {training_metrics['cv_folds']}-fold checks on the training split; "
        f"holdout evaluation on {len(test_scores):,} separate synthetic profiles. "
        "These metrics do not predict real customer behavior."
    )
    score_columns = st.columns(5)
    for column, label, key in zip(
        score_columns,
        ["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"],
        ["precision", "recall", "f1", "roc_auc", "pr_auc"],
    ):
        column.metric(label, f"{holdout_metrics[key]:.3f}")
    confusion = pd.DataFrame(
        holdout_metrics["confusion_matrix"],
        index=["Actual demo active", "Actual demo churn"],
        columns=["Predicted demo active", "Predicted demo churn"],
    )
    with st.container(border=True):
        st.markdown(
            f'<div class="ci-card-title">Synthetic holdout confusion matrix · threshold {threshold:.2f}</div>'
            '<div class="ci-card-copy">Illustrative predictions compared with synthetic holdout labels.</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(confusion, width="stretch")

    scored = test_scores.copy()
    scored["risk_band"] = pd.cut(
        scored["predicted_demo_churn_probability"],
        bins=[-0.01, 0.35, 0.65, 1.0],
        labels=["Lower", "Moderate", "Higher"],
    ).astype(str)
    scored["data_classification"] = "DEMO DATA — NOT REAL CUSTOMERS"
    scored = scored.sort_values("predicted_demo_churn_probability", ascending=False)
    with st.container(border=True):
        st.markdown(
            '<div class="ci-card-title">Scored synthetic holdout customers</div>'
            '<div class="ci-card-copy">Highest illustrative probabilities first; actual demo outcome is shown for validation only.</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(
            scored.head(12),
            hide_index=True,
            width="stretch",
            column_config={
                "predicted_demo_churn_probability": st.column_config.NumberColumn(
                    "Predicted demo churn probability", format="percent"
                ),
            },
        )
    st.download_button(
        "Download scored synthetic holdout",
        data=scored.to_csv(index=False).encode("utf-8-sig"),
        file_name="demo_scored_customers_DEMO_DATA_NOT_REAL_CUSTOMERS.csv",
        mime="text/csv",
    )


def _render_churn_analysis():
    if is_demo:
        _render_demo_churn()
        return
    st.markdown(
        '<div class="ci-status-note">This page analyzes labels in the uploaded historical CSV. A cancellation request is not a churn label. '
        'Cross-validation is retrospective and does not guarantee future performance.</div>',
        unsafe_allow_html=True,
    )
    if source_error:
        st.error(source_error)
    if source_df is None:
        st.info("Upload a historical customer CSV from the sidebar to begin.")
        return
    target = st.session_state.get("confirmed_target_column")
    if not isinstance(target, str) or target not in source_df.columns:
        st.warning("Confirm the historical target mapping in **Data Quality/Settings** before analyzing or training.")
        return
    mapping_is_current = (
        st.session_state.get("target_mapping_confirmed", False)
        and st.session_state.get("confirmed_dataset_key") == dataset_key
    )
    if not mapping_is_current:
        st.warning("Confirm the historical target mapping in **Data Quality/Settings** for this uploaded file before analyzing or training.")
        return

    data, unmapped = convert_churn_target(source_df, target, return_unmapped=True)
    if unmapped:
        st.warning(f"These non-empty target values were not mapped and are excluded from model evaluation: {unmapped}")
    labeled = data.loc[data["churn_target"].notna()].copy()
    if labeled.empty:
        st.error("No target values could be mapped to churn (1) or active (0). Check the target mapping.")
        return
    labeled["churn_target"] = labeled["churn_target"].astype(int)
    positives = int((labeled["churn_target"] == 1).sum())
    total = len(labeled)
    cols = st.columns(3)
    with cols[0].container(border=True):
        st.metric("Labeled historical customers", f"{total:,}")
    with cols[1].container(border=True):
        st.metric("Historical churn labels", f"{positives:,}")
    with cols[2].container(border=True):
        st.metric("Historical churn rate", f"{positives / total:.1%}")
    st.caption("A historical churn label describes an outcome in the supplied dataset; it is not an active cancellation request.")

    choices = [name for name in ("contract", "plan", "subscription", "state") if name in mapping]
    if choices:
        selected_categories = st.multiselect("Break down historical churn by", choices, default=choices[:2])
        category_cols = st.columns(max(1, min(2, len(selected_categories))))
        for index, standard_name in enumerate(selected_categories):
            column_name = mapping[standard_name]
            table = churn_by_category(labeled, column_name, "churn_target")
            if table is None:
                continue
            plot_data = table.rename(columns={0: "Active", 1: "Historical churn"}).reset_index()
            for label in ["Active", "Historical churn"]:
                if label not in plot_data.columns:
                    plot_data[label] = 0
            with category_cols[index % len(category_cols)]:
                with category_cols[index % len(category_cols)].container(border=True):
                    st.markdown(f'<div class="ci-card-title">Historical labels by {standard_name.replace("_", " ")}</div>', unsafe_allow_html=True)
                    chart = px.bar(
                        plot_data,
                        x=column_name,
                        y=["Active", "Historical churn"],
                        barmode="group",
                        labels={column_name: standard_name.title(), "value": "Customers", "variable": "Historical label"},
                        color_discrete_map={"Active": "#A9D8F5", "Historical churn": "#4C9ED0"},
                    )
                    st.plotly_chart(_style_chart(chart), width="stretch")
    else:
        st.info("No plan, contract, subscription, or region columns were detected for segment charts.")

    st.divider()
    st.markdown("#### Model evaluation")
    st.write("IDs and likely post-outcome leakage fields are excluded automatically. Review the exact eligible inputs before training.")
    features, excluded = eligible_feature_columns(labeled, "churn_target")
    with st.expander(f"Exact model inputs ({len(features)} features) and exclusions"):
        st.markdown("**Features used**")
        st.code(", ".join(map(str, features)) if features else "None", language=None)
        st.markdown("**Excluded columns and reasons**")
        st.write(excluded if excluded else "None")
    threshold = st.slider("Decision threshold", min_value=0.05, max_value=0.95, value=0.50, step=0.05, help="Higher thresholds usually classify fewer customers as high risk. The selected threshold changes precision, recall, F1, and the confusion matrix.")
    if st.button("Train / refresh model", type="primary"):
        try:
            fitted_model, metrics = train_model(labeled, threshold=threshold)
        except ValueError as error:
            st.error(str(error))
        except (TypeError, KeyError) as error:
            st.error(f"The uploaded data is incompatible with model training: {error}")
        else:
            st.session_state["trained_model"] = fitted_model
            st.session_state["training_metrics"] = metrics
            st.session_state["trained_dataset_key"] = dataset_key
            st.success("Model trained. Evaluation below uses stratified out-of-fold predictions, not training-set predictions.")

    valid_model = (
        st.session_state.get("trained_model") is not None
        and st.session_state.get("trained_dataset_key") == dataset_key
    )
    if not valid_model:
        st.info("Train a model to view validation metrics and score a separate active-customer file.")
        return

    metrics = st.session_state["training_metrics"]
    threshold_metrics = evaluate_threshold(
        metrics["validation_targets"],
        metrics["validation_probabilities"],
        threshold,
    )
    st.caption(f"Validation: {metrics['validation']} with {metrics['cv_folds']} stratified folds, {metrics['sample_count']:,} labeled rows.")
    metric_cols = st.columns(5)
    for column, label, key in zip(
        metric_cols,
        ["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"],
        ["precision", "recall", "f1", "roc_auc", "pr_auc"],
    ):
        with column.container(border=True):
            column.metric(label, f"{threshold_metrics[key]:.3f}")
    confusion = pd.DataFrame(
        threshold_metrics["confusion_matrix"],
        index=["Actual active (0)", "Actual churn (1)"],
        columns=["Predicted active (0)", "Predicted churn (1)"],
    )
    with st.container(border=True):
        st.markdown(f'<div class="ci-card-title">Confusion matrix · threshold {threshold:.2f}</div><div class="ci-card-copy">Predicted classes compared with historical labels.</div>', unsafe_allow_html=True)
        st.dataframe(confusion, width="stretch")
    st.caption("Changing the threshold affects precision, recall, F1, and the confusion matrix; it does not change the underlying risk probabilities.")

    st.markdown("#### Score eligible active customers")
    active_file = st.file_uploader(
        "Upload a separate CSV containing active customers only",
        type=["csv"],
        key="active_customer_csv",
        help="Do not upload cancellation feedback or post-cancellation fields. Feature column names must match the training file.",
    )
    if active_file is None:
        st.info("Scoring uses a separate active-customer file to avoid presenting in-sample historical scores as future predictions.")
        return
    try:
        active_customers = clean_data(load_data(active_file))
        scores = score_customers(st.session_state["trained_model"], active_customers, metrics["feature_columns"])
    except (ValueError, TypeError, KeyError) as error:
        st.error(f"Could not score this file: {error}")
        return
    active_mapping = detect_columns(active_customers)
    customer_id_column = active_mapping.get("customer_id")
    prioritized = pd.DataFrame(index=active_customers.index)
    if customer_id_column:
        prioritized["customer_id"] = active_customers[customer_id_column]
    for standard, output_name in [("plan", "plan"), ("monthly_charges", "monthly_charge")]:
        column_name = active_mapping.get(standard)
        if column_name and column_name in active_customers.columns:
            prioritized[output_name] = active_customers[column_name]
    prioritized["estimated_risk_score"] = scores
    prioritized = prioritized.sort_values("estimated_risk_score", ascending=False)
    with st.container(border=True):
        st.markdown('<div class="ci-status-note">Risk scores are estimates, not certainties. Historical evaluation does not guarantee future performance. Score a separate file of active customers, not cancellation feedback or post-cancellation records.</div>', unsafe_allow_html=True)
        st.dataframe(prioritized.head(100), hide_index=True, width="stretch")
        st.download_button(
            "Export prioritized retention list",
            data=prioritized.to_csv(index=False).encode("utf-8-sig"),
            file_name="prioritized_retention_list.csv",
            mime="text/csv",
        )


if page == "Welcome":
    _render_welcome()
elif page == "Overview":
    _render_overview()
elif page == "New Cancellation Review":
    _render_new_case()
elif page == "Case History":
    _render_case_history()
elif page == "Churn Analysis":
    _render_churn_analysis()
else:
    _render_quality()
