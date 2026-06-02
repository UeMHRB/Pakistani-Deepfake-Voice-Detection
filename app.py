import streamlit as st
import librosa
import numpy as np
import pickle
import os
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import noisereduce as nr
from scipy.ndimage import gaussian_filter1d
from datetime import datetime
import tempfile
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Image, Table, TableStyle, HRFlowable)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title = 'Deepfake Voice Detector',
    page_icon  = '🎙️',
    layout     = 'wide'
)

# ── Load model and scaler ─────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'audio_reports')
CHARTS_DIR  = os.path.join(BASE_DIR, 'audio_reports', 'charts')

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR,  exist_ok=True)

@st.cache_resource
def load_model():
    model      = pickle.load(open(os.path.join(MODELS_DIR, 'model.pkl'),      'rb'))
    scaler     = pickle.load(open(os.path.join(MODELS_DIR, 'scaler.pkl'),     'rb'))
    real_means = np.load(os.path.join(MODELS_DIR, 'real_means.npy'))
    real_stds  = np.load(os.path.join(MODELS_DIR, 'real_stds.npy'))
    return model, scaler, real_means, real_stds

model, scaler, real_means, real_stds = load_model()

# ── MFCC names ────────────────────────────────────────────────────────────
MFCC_NAMES = {
    'MFCC_1'  : 'Overall Energy Level',
    'MFCC_2'  : 'Spectral Shape (Brightness)',
    'MFCC_3'  : 'Voice Texture Coarseness',
    'MFCC_4'  : 'Vocal Tract Resonance',
    'MFCC_5'  : 'Nasal Resonance Pattern',
    'MFCC_6'  : 'Mid Frequency Tone',
    'MFCC_7'  : 'Articulation Clarity',
    'MFCC_8'  : 'Pitch Variation Pattern',
    'MFCC_9'  : 'Voice Smoothness',
    'MFCC_10' : 'Spectral Fine Structure',
    'MFCC_11' : 'High Frequency Detail',
    'MFCC_12' : 'Breathiness Level',
    'MFCC_13' : 'Consonant Sharpness',
    'MFCC_14' : 'Vowel Quality',
    'MFCC_15' : 'Rhythm Pattern',
    'MFCC_16' : 'Stress Pattern',
    'MFCC_17' : 'Speaking Rate Variation',
    'MFCC_18' : 'Voice Onset Pattern',
    'MFCC_19' : 'Intonation Contour',
    'MFCC_20' : 'Micro Pitch Fluctuation',
    'MFCC_21' : 'Spectral Tilt',
    'MFCC_22' : 'Formant Transition Speed',
    'MFCC_23' : 'Voice Naturalness',
    'MFCC_24' : 'Glottal Pulse Pattern',
    'MFCC_25' : 'Subglottal Resonance',
    'MFCC_26' : 'Palatal Resonance',
    'MFCC_27' : 'Velar Resonance',
    'MFCC_28' : 'Labial Sound Pattern',
    'MFCC_29' : 'Dental Sound Texture',
    'MFCC_30' : 'Fricative Pattern',
    'MFCC_31' : 'Plosive Burst Pattern',
    'MFCC_32' : 'Nasal Murmur Detail',
    'MFCC_33' : 'Liquid Sound Texture',
    'MFCC_34' : 'Glide Transition Pattern',
    'MFCC_35' : 'Voice Tremor Detail',
    'MFCC_36' : 'Micro Rhythm Pattern',
    'MFCC_37' : 'Spectral Envelope Shape',
    'MFCC_38' : 'Harmonic Structure',
    'MFCC_39' : 'Noise Floor Pattern',
    'MFCC_40' : 'Ultra Fine Voice Texture',
}

FAKE_RED   = '#E24B4A'
FAKE_LIGHT = '#FCEBEB'
REAL_GREEN = '#639922'
REAL_LIGHT = '#EAF3DE'
BLUE       = '#85B7EB'
GRAY       = '#888780'
DARK_GRAY  = '#444441'

# ── Core functions ────────────────────────────────────────────────────────
def clean_audio(audio, sr):
    try:
        noise_sample  = audio[:int(0.3 * sr)]
        audio_cleaned = nr.reduce_noise(
            y=audio, sr=sr, y_noise=noise_sample, prop_decrease=0.6
        )
        frame_length = 512
        hop_length   = 256
        rms          = librosa.feature.rms(
            y=audio_cleaned, frame_length=frame_length, hop_length=hop_length
        )[0]
        mean_rms  = np.mean(rms)
        threshold = mean_rms + 2 * np.std(rms)
        gain = np.ones(len(rms))
        for i, r in enumerate(rms):
            if r > threshold:
                gain[i] = mean_rms / r
        gain_smooth = gaussian_filter1d(gain, sigma=5)
        gain_full   = np.interp(
            np.arange(len(audio_cleaned)),
            np.arange(len(rms)) * hop_length,
            gain_smooth
        )
        audio_cleaned = audio_cleaned * gain_full
        max_amp = np.max(np.abs(audio_cleaned))
        if max_amp > 0:
            audio_cleaned = audio_cleaned / max_amp * 0.9
        return audio_cleaned
    except:
        return audio

def predict_audio(audio, sr):
    audio_cleaned = clean_audio(audio, sr)
    chunk_size    = 3 * 16000
    total_chunks  = len(audio_cleaned) // chunk_size

    if total_chunks == 0:
        audio_cleaned = np.pad(
            audio_cleaned, (0, chunk_size - len(audio_cleaned)), mode='constant'
        )
        total_chunks = 1

    chunk_results = []
    for i in range(total_chunks):
        chunk           = audio_cleaned[i*chunk_size:(i+1)*chunk_size]
        mfcc            = librosa.feature.mfcc(y=chunk, sr=16000, n_mfcc=40)
        features        = np.mean(mfcc, axis=1).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction      = model.predict(features_scaled)[0]
        confidence      = model.predict_proba(features_scaled)[0]
        conf_pct        = confidence[prediction] * 100
        label           = 'FAKE' if prediction == 1 else 'REAL'
        chunk_results.append({
            'chunk': i+1, 'start': i*3, 'end': (i+1)*3,
            'label': label, 'confidence': conf_pct
        })

    fake_count = sum(1 for r in chunk_results if r['label'] == 'FAKE')
    real_count = sum(1 for r in chunk_results if r['label'] == 'REAL')
    avg_conf   = sum(r['confidence'] for r in chunk_results) / len(chunk_results)
    final      = 'FAKE' if fake_count > real_count else 'REAL'

    return chunk_results, final, avg_conf

def generate_timeline_chart(results, timestamp):
    fig, ax = plt.subplots(figsize=(10, 2.5))
    fig.patch.set_facecolor('white')
    for r in results:
        color     = FAKE_RED   if r['label'] == 'FAKE' else REAL_GREEN
        edgecolor = '#A32D2D'  if r['label'] == 'FAKE' else '#27500A'
        ax.barh(y=0, width=3, left=r['start'], color=color,
                alpha=0.85, height=0.6, edgecolor=edgecolor, linewidth=1.5)
        ax.text(r['start']+1.5,  0.18, r['label'],
                ha='center', va='center', fontsize=10,
                color='white', fontweight='bold')
        ax.text(r['start']+1.5, -0.15, f"{r['confidence']:.0f}% confident",
                ha='center', va='center', fontsize=9, color='white')
    total_duration = results[-1]['end']
    for r in results:
        ax.text(r['start'], -0.42, f"{r['start']}s",
                ha='center', fontsize=9, color=DARK_GRAY)
    ax.text(total_duration, -0.42, f"{total_duration}s",
            ha='center', fontsize=9, color=DARK_GRAY)
    fake_patch = mpatches.Patch(color=FAKE_RED,   label='FAKE — AI Generated')
    real_patch = mpatches.Patch(color=REAL_GREEN, label='REAL — Human Voice')
    ax.legend(handles=[fake_patch, real_patch], loc='upper right', fontsize=9)
    ax.set_xlim(-0.2, total_duration + 0.2)
    ax.set_ylim(-0.55, 0.55)
    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title('Each block = 3 seconds. Color shows verdict for that segment.',
                 fontsize=9, color=GRAY, pad=8)
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, f'timeline_{timestamp}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    return path

def generate_shap_chart(audio, chunk_results, final, timestamp):
    fake_chunks = [r for r in chunk_results if r['label'] == 'FAKE']
    target      = fake_chunks[0] if fake_chunks else chunk_results[0]

    audio_cleaned = clean_audio(audio, 16000)
    start         = target['start'] * 16000
    chunk         = audio_cleaned[start:start + 3*16000]
    mfcc          = librosa.feature.mfcc(y=chunk, sr=16000, n_mfcc=40)
    features      = np.mean(mfcc, axis=1).reshape(1, -1)
    features_scaled = scaler.transform(features)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features_scaled)

    if isinstance(shap_values, list):
        vals = np.array(shap_values[1][0]).flatten()
    elif hasattr(shap_values, 'values'):
        vals = np.array(shap_values.values).flatten()[:40]
    else:
        vals = np.array(shap_values).flatten()[:40]
    vals = vals[:40]

    feature_names = [f'MFCC_{i+1}' for i in range(40)]
    indices       = np.argsort(np.abs(vals))[-12:]
    indices       = [int(i) for i in indices]
    top_vals      = [vals[i] for i in indices]
    top_keys      = [feature_names[i] for i in indices]
    top_names     = [f"{MFCC_NAMES.get(k, k)}\n({k})" for k in top_keys]
    bar_colors    = [FAKE_RED if v > 0 else REAL_GREEN for v in top_vals]

    norm_lo = [real_means[i] - real_stds[i] for i in indices]
    norm_hi = [real_means[i] + real_stds[i] for i in indices]
    actual  = [features_scaled[0][i] for i in indices]

    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor('white')
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.45)

    ax1 = fig.add_subplot(gs[0])
    bars = ax1.barh(top_names, top_vals, color=bar_colors, alpha=0.85,
                    edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, top_vals):
        x = bar.get_width()
        ax1.text(x + (0.001 if x >= 0 else -0.001),
                 bar.get_y() + bar.get_height()/2,
                 f'{"+" if val > 0 else ""}{val:.3f}',
                 va='center', ha='left' if x >= 0 else 'right',
                 fontsize=8, color=FAKE_RED if val > 0 else REAL_GREEN)
    ax1.axvline(x=0, color=DARK_GRAY, linewidth=1)
    ax1.set_xlabel('SHAP Value — how strongly this feature influences the decision',
                   fontsize=9, color=GRAY)
    title_word = 'FAKE' if final == 'FAKE' else 'REAL'
    ax1.set_title(f'Why is it {title_word}?\nTop contributing voice features',
                  fontsize=11, fontweight='bold', pad=12)
    ax1.tick_params(axis='y', labelsize=8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    fake_patch = mpatches.Patch(color=FAKE_RED,   label='Pushing toward FAKE')
    real_patch = mpatches.Patch(color=REAL_GREEN, label='Pushing toward REAL')
    ax1.legend(handles=[fake_patch, real_patch], fontsize=8, loc='lower right')

    ax2   = fig.add_subplot(gs[1])
    y_pos = range(len(top_names))
    for j, (lo, hi) in enumerate(zip(norm_lo, norm_hi)):
        ax2.barh(j, hi-lo, left=lo, color=BLUE, alpha=0.35, height=0.5,
                 label='Normal range' if j == 0 else '')
    for j, (act, val) in enumerate(zip(actual, top_vals)):
        inside    = norm_lo[j] <= act <= norm_hi[j]
        dot_color = REAL_GREEN if inside else FAKE_RED
        ax2.scatter(act, j, color=dot_color, zorder=5, s=100,
                    edgecolors='white', linewidth=1)
        status       = 'Within range' if inside else 'Outside range'
        status_color = REAL_GREEN if inside else FAKE_RED
        xlim = ax2.get_xlim()
        ax2.text(xlim[1] + 0.1, j, f'  {status}',
                 va='center', fontsize=7.5, color=status_color)
    ax2.set_yticks(list(y_pos))
    ax2.set_yticklabels(top_names, fontsize=8)
    ax2.set_xlabel('Feature value (normalized). Blue = normal human range.',
                   fontsize=9, color=GRAY)
    ax2.set_title('Normal Range vs This Audio\nDot outside blue bar = suspicious',
                  fontsize=11, fontweight='bold', pad=12)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    outside_patch = mpatches.Patch(color=FAKE_RED,   label='Outside normal range')
    inside_patch  = mpatches.Patch(color=REAL_GREEN, label='Within normal range')
    range_patch   = mpatches.Patch(color=BLUE, alpha=0.5, label='Normal human range')
    ax2.legend(handles=[range_patch, inside_patch, outside_patch],
               fontsize=8, loc='lower right')

    plt.suptitle('Voice Analysis — Feature Explanation',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, f'shap_{timestamp}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    return path

def generate_pdf(filename, results, final, avg_conf,
                 timeline_path, shap_path, timestamp):

    pdf_path = os.path.join(REPORTS_DIR, f'{filename}_{timestamp}.pdf')
    doc      = SimpleDocTemplate(pdf_path, pagesize=A4,
                                  topMargin=0.5*inch, bottomMargin=0.5*inch,
                                  leftMargin=0.6*inch, rightMargin=0.6*inch)
    styles = getSampleStyleSheet()
    story  = []

    title_style = ParagraphStyle('T', parent=styles['Title'],
                                  fontSize=22, spaceAfter=4,
                                  textColor=colors.HexColor(DARK_GRAY))
    sub_style   = ParagraphStyle('S', parent=styles['Normal'],
                                  fontSize=10, textColor=colors.HexColor(GRAY),
                                  spaceAfter=16)
    section_style = ParagraphStyle('H', parent=styles['Heading2'],
                                    fontSize=13, spaceBefore=14, spaceAfter=6)
    explain_style = ParagraphStyle('E', parent=styles['Normal'],
                                    fontSize=9, textColor=colors.HexColor(GRAY),
                                    backColor=colors.HexColor('#F1EFE8'),
                                    borderPad=8, spaceAfter=10,
                                    leftIndent=8, rightIndent=8)

    story.append(Paragraph('Deepfake Voice Detection Report', title_style))
    story.append(Paragraph(
        f'Pakistani Urdu-English Mixed Audio Analysis · '
        f'Generated {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        sub_style
    ))
    story.append(HRFlowable(width='100%', thickness=1,
                             color=colors.HexColor('#D3D1C7')))
    story.append(Spacer(1, 0.15*inch))

    if final == 'FAKE':
        banner_bg  = colors.HexColor(FAKE_LIGHT)
        banner_bdr = colors.HexColor(FAKE_RED)
        v_color    = colors.HexColor('#791F1F')
        conf_bg    = colors.HexColor('#F09595')
        v_text     = 'AI Generated Voice Detected'
        v_sub      = 'This audio is likely synthetic — not a real human voice'
    else:
        banner_bg  = colors.HexColor(REAL_LIGHT)
        banner_bdr = colors.HexColor(REAL_GREEN)
        v_color    = colors.HexColor('#27500A')
        conf_bg    = colors.HexColor('#C0DD97')
        v_text     = 'Real Human Voice Confirmed'
        v_sub      = 'This audio appears to be a genuine human voice'

    fake_count = sum(1 for r in results if r['label'] == 'FAKE')
    real_count = sum(1 for r in results if r['label'] == 'REAL')

    verdict_data = [[
        Paragraph(f'<b>{v_text}</b>',
                  ParagraphStyle('VT', fontSize=16, textColor=v_color, spaceAfter=4)),
        Paragraph(f'<b>{avg_conf:.1f}%</b><br/>Confidence',
                  ParagraphStyle('VC', fontSize=14, textColor=v_color,
                                 alignment=TA_CENTER))
    ],[
        Paragraph(v_sub, ParagraphStyle('VS', fontSize=9,
                  textColor=colors.HexColor('#3B6D11' if final=='REAL' else '#A32D2D'))),
        Paragraph(f'File: {filename}',
                  ParagraphStyle('VF', fontSize=8,
                                 textColor=colors.HexColor(GRAY),
                                 alignment=TA_CENTER))
    ]]
    vt = Table(verdict_data, colWidths=[4.5*inch, 1.8*inch])
    vt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), banner_bg),
        ('LINEABOVE',  (0,0), (-1,0),  4, banner_bdr),
        ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING',    (0,0), (-1,-1), 12),
        ('BACKGROUND', (1,0), (1,-1),  conf_bg),
    ]))
    story.append(vt)
    story.append(Spacer(1, 0.15*inch))

    stats_data = [[
        Paragraph(f'<b>Total Segments</b><br/>{len(results)}',
                  ParagraphStyle('S1', fontSize=11, alignment=TA_CENTER,
                                 textColor=colors.HexColor(DARK_GRAY))),
        Paragraph(f'<b>Fake Segments</b><br/>'
                  f'<font color="{FAKE_RED}">{fake_count}</font>',
                  ParagraphStyle('S2', fontSize=11, alignment=TA_CENTER,
                                 textColor=colors.HexColor(DARK_GRAY))),
        Paragraph(f'<b>Real Segments</b><br/>'
                  f'<font color="{REAL_GREEN}">{real_count}</font>',
                  ParagraphStyle('S3', fontSize=11, alignment=TA_CENTER,
                                 textColor=colors.HexColor(DARK_GRAY))),
        Paragraph(f'<b>Audio Duration</b><br/>{results[-1]["end"]}s',
                  ParagraphStyle('S4', fontSize=11, alignment=TA_CENTER,
                                 textColor=colors.HexColor(DARK_GRAY))),
    ]]
    st_table = Table(stats_data, colWidths=[1.575*inch]*4)
    st_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F1EFE8')),
        ('GRID',       (0,0), (-1,-1), 0.5, colors.HexColor('#D3D1C7')),
        ('PADDING',    (0,0), (-1,-1), 10),
        ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
        ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(st_table)
    story.append(Spacer(1, 0.1*inch))

    story.append(Paragraph('Timeline — Which Seconds Are Suspicious?', section_style))
    story.append(Paragraph(
        'Each block = 3 seconds. Red = AI Generated. Green = Human Voice. '
        'Percentage = model confidence for that segment.',
        explain_style
    ))
    story.append(Image(timeline_path, width=6.3*inch, height=1.8*inch))
    story.append(Spacer(1, 0.1*inch))

    title_word = 'FAKE' if final == 'FAKE' else 'REAL'
    story.append(Paragraph(f'Voice Feature Analysis — Why is it {title_word}?',
                            section_style))
    story.append(Paragraph(
        'LEFT: Red bars push toward FAKE, Green toward REAL. '
        'RIGHT: Blue bar = normal human voice range. '
        'Dot outside blue = suspicious feature.',
        explain_style
    ))
    story.append(Image(shap_path, width=6.3*inch, height=4.2*inch))
    story.append(Spacer(1, 0.1*inch))

    story.append(Paragraph('Segment-by-Segment Breakdown', section_style))
    chunk_data = [[
        Paragraph('<b>Segment</b>', styles['Normal']),
        Paragraph('<b>Time</b>',    styles['Normal']),
        Paragraph('<b>Result</b>',  styles['Normal']),
        Paragraph('<b>Confidence</b>', styles['Normal']),
        Paragraph('<b>Suspicion Level</b>', styles['Normal']),
    ]]
    for r in results:
        is_fake  = r['label'] == 'FAKE'
        res_color = FAKE_RED   if is_fake else REAL_GREEN
        bar_filled = int(r['confidence'] / 100 * 20)
        bar_str    = '█' * bar_filled + '░' * (20 - bar_filled)
        chunk_data.append([
            Paragraph(str(r['chunk']),    styles['Normal']),
            Paragraph(f"{r['start']}s — {r['end']}s", styles['Normal']),
            Paragraph(f'<font color="{res_color}"><b>{r["label"]}</b></font>',
                      styles['Normal']),
            Paragraph(f"{r['confidence']:.1f}%", styles['Normal']),
            Paragraph(f'<font color="{res_color}" size="7">{bar_str}</font>',
                      styles['Normal']),
        ])
    ct = Table(chunk_data, colWidths=[0.7*inch, 1.1*inch, 0.9*inch,
                                       1*inch, 2.6*inch])
    ct.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0),  colors.HexColor(DARK_GRAY)),
        ('TEXTCOLOR',     (0,0), (-1,0),  colors.white),
        ('FONTNAME',      (0,0), (-1,0),  'Helvetica-Bold'),
        ('ROWBACKGROUNDS',(0,1), (-1,-1),
         [colors.white, colors.HexColor('#F1EFE8')]),
        ('GRID',          (0,0), (-1,-1), 0.5, colors.HexColor('#D3D1C7')),
        ('PADDING',       (0,0), (-1,-1), 8),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(ct)
    story.append(Spacer(1, 0.3*inch))
    story.append(HRFlowable(width='100%', thickness=0.5,
                             color=colors.HexColor('#D3D1C7')))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph(
        'Generated by Deepfake Voice Detection System — '
        'Pakistani Urdu-English Code-Switched Audio | '
        'Model: Random Forest | Accuracy: 95.4% | Features: MFCC (40)',
        ParagraphStyle('F', fontSize=8, textColor=colors.HexColor(GRAY),
                       alignment=TA_CENTER)
    ))
    doc.build(story)
    return pdf_path

# ── Streamlit UI ──────────────────────────────────────────────────────────
st.title('🎙️ Deepfake Voice Detection')
st.markdown('**Pakistani Urdu-English Code-Switched Audio Analysis**')
st.markdown('---')

# Sidebar
with st.sidebar:
    st.header('About')
    st.markdown('''
    This system detects whether a voice is:
    - **Real** — genuine human voice
    - **Fake** — AI generated voice

    **How it works:**
    1. Audio is cleaned and split into 3 second chunks
    2. MFCC features extracted from each chunk
    3. Random Forest model predicts each chunk
    4. Majority vote gives final verdict

    **Model accuracy:** 95.4%

    **Supported languages:**
    Urdu, English, and code-switched Urdu+English
    ''')
    st.markdown('---')
    st.markdown('NAVTTC AI/ML Program · NETSOL Institute')

# Main content
uploaded_file = st.file_uploader(
    'Upload an audio file to analyze',
    type=['wav', 'mp3'],
    help='Supported formats: WAV, MP3'
)

if uploaded_file is not None:
    st.audio(uploaded_file)
    st.markdown('---')

    with st.spinner('Analyzing audio — please wait...'):

        # Save to temp file
        suffix = '.wav' if uploaded_file.name.endswith('.wav') else '.mp3'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(uploaded_file.read())
            temp_path = f.name

        # Load audio
        audio, sr = librosa.load(temp_path, sr=16000, mono=True)

        # Predict
        results, final, avg_conf = predict_audio(audio, sr)

        # Generate charts
        timestamp      = datetime.now().strftime('%Y%m%d_%H%M%S')
        audio_name     = os.path.splitext(uploaded_file.name)[0]
        timeline_path  = generate_timeline_chart(results, timestamp)
        shap_path      = generate_shap_chart(audio, results, final, timestamp)

        # Generate PDF
        pdf_path = generate_pdf(
            audio_name, results, final, avg_conf,
            timeline_path, shap_path, timestamp
        )

        os.unlink(temp_path)

    # ── Verdict banner ────────────────────────────────────────────────────
    st.markdown('## Analysis Result')

    if final == 'FAKE':
        st.error(f'### 🚨 AI Generated Voice Detected')
        col1, col2, col3, col4 = st.columns(4)
        col1.metric('Final Result',      'FAKE')
        col2.metric('Confidence',        f'{avg_conf:.1f}%')
        col3.metric('Fake Segments',     sum(1 for r in results if r['label']=='FAKE'))
        col4.metric('Total Segments',    len(results))
    else:
        st.success(f'### ✅ Real Human Voice Confirmed')
        col1, col2, col3, col4 = st.columns(4)
        col1.metric('Final Result',      'REAL')
        col2.metric('Confidence',        f'{avg_conf:.1f}%')
        col3.metric('Real Segments',     sum(1 for r in results if r['label']=='REAL'))
        col4.metric('Total Segments',    len(results))

    st.markdown('---')

    # ── Timeline ──────────────────────────────────────────────────────────
    st.markdown('### Timeline — Which Seconds Are Suspicious?')
    st.caption('Each block = 3 seconds. Red = Fake. Green = Real. '
               'Percentage shows model confidence.')
    st.image(timeline_path, use_column_width=True)

    st.markdown('---')

    # ── SHAP chart ────────────────────────────────────────────────────────
    title_word = 'FAKE' if final == 'FAKE' else 'REAL'
    st.markdown(f'### Voice Feature Analysis — Why is it {title_word}?')
    st.caption(
        'LEFT: Red bars push toward FAKE, Green toward REAL. '
        'RIGHT: Blue bar = normal human range. Dot outside = suspicious.'
    )
    st.image(shap_path, use_column_width=True)

    st.markdown('---')

    # ── Chunk table ───────────────────────────────────────────────────────
    st.markdown('### Segment-by-Segment Breakdown')
    for r in results:
        col1, col2, col3, col4 = st.columns([1, 2, 2, 3])
        col1.write(f"**{r['chunk']}**")
        col2.write(f"{r['start']}s — {r['end']}s")
        if r['label'] == 'FAKE':
            col3.error(f"FAKE")
        else:
            col3.success(f"REAL")
        col4.progress(int(r['confidence']), text=f"{r['confidence']:.1f}%")

    st.markdown('---')

    # ── Download PDF ──────────────────────────────────────────────────────
    st.markdown('### Download Full Report')
    with open(pdf_path, 'rb') as f:
        st.download_button(
            label     = '📄 Download PDF Report',
            data      = f,
            file_name = f'{audio_name}_{timestamp}.pdf',
            mime      = 'application/pdf'
        )