#!/usr/bin/env python3
"""
Interactive Streamlit dashboard for the AI Candidate Ranking Engine.
Allows judges to:
  - View ranked shortlist with score breakdowns
  - See per-candidate rationale and radar/bar charts
  - Toggle scoring signals on/off to see impact
  - Give thumbs up/down feedback (feedback loop)
  - View ablation study results

Launch: streamlit run app/dashboard.py
"""

import json
import os
import sys
import csv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ensure project root is on path
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# =========================================================================
# Page config
# =========================================================================
st.set_page_config(
    page_title="AI Candidate Ranking Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================================
# Custom CSS for premium feel
# =========================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    .main { 
        background: radial-gradient(circle at top right, #111424 0%, #080A10 100%);
    }
    
    /* Premium Headers */
    h1 { 
        background: -webkit-linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        margin-bottom: 0px !important;
    }
    h2, h3 { color: #E2E8F0; font-weight: 600; }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 700;
        color: #4ECDC4;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        color: #A0AEC0;
        font-weight: 400;
    }
    div[data-testid="metric-container"] {
        background: rgba(30, 33, 48, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 20px;
        border-radius: 16px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        border-color: #4ECDC4;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(20, 24, 39, 0.8);
        padding: 5px;
        border-radius: 12px;
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        color: #A0AEC0;
        font-weight: 600;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #FFFFFF;
        background: rgba(255,255,255,0.05);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4ECDC4 0%, #2B6CB0 100%) !important;
        color: white !important;
    }
    
    /* Cards (st.container with border) */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid rgba(255,255,255,0.08) !important;
        background: rgba(25, 28, 43, 0.4) !important;
        border-radius: 16px !important;
        padding: 10px 15px !important;
        transition: all 0.3s ease;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        background: rgba(30, 35, 55, 0.7) !important;
        border-color: rgba(78, 205, 196, 0.5) !important;
        box-shadow: 0 8px 30px rgba(0,0,0,0.3);
    }
    
    /* Progress Bars */
    .stProgress .st-bo {
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
    }
    
    /* Buttons */
    .stButton button {
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.1);
        background: rgba(255,255,255,0.05);
        transition: all 0.2s ease;
    }
    .stButton button:hover {
        background: rgba(255,255,255,0.15);
        border-color: #4ECDC4;
        transform: scale(1.05);
    }
    
    /* Text improvements */
    p { color: #CBD5E0; }
    strong { color: #FFFFFF; }
</style>
""", unsafe_allow_html=True)


# =========================================================================
# Helper functions
# =========================================================================
@st.cache_data
def load_results():
    """Load ranking results from output files."""
    detailed_path = os.path.join(PROJECT_ROOT, "output", "submission_detailed.json")
    csv_path = os.path.join(PROJECT_ROOT, "output", "submission.csv")
    
    # Try detailed JSON first
    if os.path.exists(detailed_path):
        with open(detailed_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    # Fallback to CSV
    if os.path.exists(csv_path):
        results = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append({
                    "candidate_id": row["candidate_id"],
                    "rank": int(row["rank"]),
                    "final_score": float(row["score"]),
                    "reasoning": row.get("reasoning", ""),
                })
        return results
    
    return None


@st.cache_data
def load_candidates():
    """Load candidate profiles for detail view."""
    sample_path = os.path.join(PROJECT_ROOT, "data", "sample_candidates.json")
    if os.path.exists(sample_path):
        with open(sample_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {c["candidate_id"]: c for c in data}
    return {}


def create_radar_chart(scores: dict, title: str = "Score Breakdown"):
    """Create a radar chart for score visualization."""
    categories = list(scores.keys())
    values = list(scores.values())
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(99, 110, 250, 0.2)',
        line=dict(color='#636EFA', width=2),
        name='Scores',
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
            bgcolor='rgba(0,0,0,0)',
        ),
        showlegend=False,
        title=title,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0ff'),
        height=350,
        margin=dict(t=50, b=30, l=50, r=50),
    )
    return fig


def create_bar_chart(scores: dict, title: str = "Sub-scores"):
    """Create a horizontal bar chart for scores."""
    fig = go.Figure()
    
    categories = list(scores.keys())
    values = list(scores.values())
    
    colors = px.colors.qualitative.Set2[:len(categories)]
    
    fig.add_trace(go.Bar(
        y=categories,
        x=values,
        orientation='h',
        marker_color=colors,
        text=[f'{v:.3f}' for v in values],
        textposition='outside',
    ))
    
    fig.update_layout(
        title=title,
        xaxis=dict(range=[0, 1.1], title="Score"),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0ff'),
        height=max(200, len(categories) * 35 + 80),
        margin=dict(t=50, b=30, l=120, r=50),
    )
    return fig


# =========================================================================
# Main dashboard
# =========================================================================
def main():
    st.title("🎯 AI Candidate Ranking Engine")
    st.markdown("*Intelligent candidate discovery using hybrid retrieval, behavioral signals, and explainable scoring*")
    
    # Load data
    results = load_results()
    candidates = load_candidates()
    
    if not results:
        st.warning("⚠️ No ranking results found. Run `python rank.py` first to generate results.")
        st.code("python rank.py --candidates data/sample_candidates.json --out output/submission.csv")
        
        # Still show the JD
        st.subheader("📋 Job Description")
        jd_path = os.path.join(PROJECT_ROOT, "data", "job_description.txt")
        if os.path.exists(jd_path):
            with open(jd_path, "r") as f:
                st.text(f.read())
        return
    
    # Sidebar: Signal weight controls (feedback loop)
    st.sidebar.header("🎛️ Signal Weights")
    st.sidebar.markdown("Adjust weights to see ranking impact:")
    
    w_semantic = st.sidebar.slider("Semantic Score", 0.0, 1.0, 0.30, 0.05)
    w_attribute = st.sidebar.slider("Attribute Score", 0.0, 1.0, 0.45, 0.05)
    w_behavioral = st.sidebar.slider("Behavioral Score", 0.0, 1.0, 0.25, 0.05)
    
    # Normalize weights
    total_w = w_semantic + w_attribute + w_behavioral
    if total_w > 0:
        w_semantic /= total_w
        w_attribute /= total_w
        w_behavioral /= total_w
    
    st.sidebar.markdown(f"**Normalized:** Sem={w_semantic:.2f} Attr={w_attribute:.2f} Beh={w_behavioral:.2f}")
    
    # Tabs
    tab_ranking, tab_detail, tab_ablation, tab_jd = st.tabs([
        "📊 Ranked Shortlist",
        "🔍 Candidate Detail",
        "📈 Ablation Study",
        "📋 Job Description",
    ])
    
    # === Tab 1: Ranked Shortlist ===
    with tab_ranking:
        st.subheader(f"Top {len(results)} Candidates")
        
        # Re-rank based on adjusted weights if detailed scores available
        display_results = results
        if results and "score_breakdown" in results[0]:
            for r in results:
                sb = r.get("score_breakdown", {})
                new_score = (w_semantic * sb.get("semantic_score", 0) +
                           w_attribute * sb.get("attribute_score", 0) +
                           w_behavioral * sb.get("behavioral_score", 0))
                r["_adjusted_score"] = new_score
            
            display_results = sorted(results, key=lambda x: -x.get("_adjusted_score", x["final_score"]))
            for i, r in enumerate(display_results):
                r["_display_rank"] = i + 1
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Candidates", len(display_results))
        with col2:
            avg_score = sum(r["final_score"] for r in display_results) / len(display_results) if display_results else 0
            st.metric("Avg Score", f"{avg_score:.4f}")
        with col3:
            top_score = display_results[0]["final_score"] if display_results else 0
            st.metric("Top Score", f"{top_score:.4f}")
        with col4:
            score_spread = (display_results[0]["final_score"] - display_results[-1]["final_score"]) if len(display_results) > 1 else 0
            st.metric("Score Spread", f"{score_spread:.4f}")
        
        # Results table
        for r in display_results[:20]:
            rank = r.get("_display_rank", r.get("rank", "?"))
            score = r.get("_adjusted_score", r["final_score"])
            
            with st.container(border=True):
                col_rank, col_info, col_score, col_feedback = st.columns([1, 5, 2, 1])
                
                with col_rank:
                    st.markdown(f"<h3 style='color: #FF6B6B; margin-top: 10px;'>#{rank}</h3>", unsafe_allow_html=True)
                
                with col_info:
                    profile = r.get("profile_summary", {})
                    title = profile.get("title", "")
                    company = profile.get("company", "")
                    exp = profile.get("experience", 0)
                    st.markdown(f"**{r['candidate_id']}** — {title} at {company} ({exp} yrs)")
                    st.caption(f"_{r.get('reasoning', '')}_")
                
                with col_score:
                    st.progress(min(score, 1.0))
                    st.caption(f"**Score:** {score:.4f}")
                
                with col_feedback:
                    # Feedback buttons (feedback loop feature)
                    key_up = f"up_{r['candidate_id']}"
                    key_down = f"down_{r['candidate_id']}"
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("👍", key=key_up, help="Mark as relevant"):
                            if "feedback" not in st.session_state:
                                st.session_state.feedback = {}
                            st.session_state.feedback[r["candidate_id"]] = "relevant"
                            st.toast(f"✅ {r['candidate_id']} marked as relevant")
                    with c2:
                        if st.button("👎", key=key_down, help="Mark as not relevant"):
                            if "feedback" not in st.session_state:
                                st.session_state.feedback = {}
                            st.session_state.feedback[r["candidate_id"]] = "not_relevant"
                            st.toast(f"❌ {r['candidate_id']} marked as not relevant")
        
        # Show feedback summary if any
        if hasattr(st.session_state, "feedback") and st.session_state.feedback:
            st.subheader("📝 Feedback Summary")
            relevant = [k for k, v in st.session_state.feedback.items() if v == "relevant"]
            not_relevant = [k for k, v in st.session_state.feedback.items() if v == "not_relevant"]
            st.success(f"👍 Relevant: {len(relevant)}")
            st.error(f"👎 Not Relevant: {len(not_relevant)}")
    
    # === Tab 2: Candidate Detail ===
    with tab_detail:
        st.subheader("🔍 Candidate Detail View")
        
        cand_ids = [r["candidate_id"] for r in results]
        selected_id = st.selectbox("Select candidate:", cand_ids)
        
        if selected_id:
            cand_result = next((r for r in results if r["candidate_id"] == selected_id), None)
            
            if cand_result:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"### {selected_id}")
                    profile = cand_result.get("profile_summary", {})
                    st.markdown(f"**{profile.get('title', '')}** at {profile.get('company', '')}")
                    st.markdown(f"📍 {profile.get('location', '')} | 📅 {profile.get('experience', 0)} years")
                    st.markdown(f"**Rank:** #{cand_result.get('rank', '?')} | **Score:** {cand_result['final_score']:.4f}")
                    st.markdown("---")
                    st.markdown("**Reasoning:**")
                    st.info(cand_result.get("reasoning", ""))
                
                with col2:
                    breakdown = cand_result.get("score_breakdown", {})
                    if breakdown:
                        # Top-level scores radar
                        top_scores = {
                            "Semantic": breakdown.get("semantic_score", 0),
                            "Attribute": breakdown.get("attribute_score", 0),
                            "Behavioral": breakdown.get("behavioral_score", 0),
                        }
                        st.plotly_chart(create_radar_chart(top_scores, "Score Components"), use_container_width=True)
                
                # Detailed breakdowns
                if breakdown:
                    col3, col4 = st.columns(2)
                    
                    with col3:
                        attr_breakdown = breakdown.get("attribute_breakdown", {})
                        if attr_breakdown:
                            st.plotly_chart(
                                create_bar_chart(attr_breakdown, "Attribute Sub-scores"),
                                use_container_width=True,
                            )
                    
                    with col4:
                        beh_breakdown = breakdown.get("behavioral_breakdown", {})
                        if beh_breakdown:
                            st.plotly_chart(
                                create_bar_chart(beh_breakdown, "Behavioral Sub-scores"),
                                use_container_width=True,
                            )
                
                # Full candidate profile
                if selected_id in candidates:
                    with st.expander("📄 Full Candidate Profile"):
                        cand_data = candidates[selected_id]
                        st.json(cand_data)
    
    # === Tab 3: Ablation Study ===
    with tab_ablation:
        st.subheader("📈 Ablation Study Results")
        st.markdown("Comparing ranking quality across different pipeline configurations.")
        
        ablation_path = os.path.join(PROJECT_ROOT, "output", "ablation_results.json")
        if os.path.exists(ablation_path):
            with open(ablation_path, "r") as f:
                ablation = json.load(f)
            
            # Build comparison table
            import pandas as pd
            rows = []
            for key, r in ablation.items():
                if "error" not in r:
                    rows.append({
                        "Method": r.get("method", key),
                        "Composite": r.get("composite", 0),
                        "NDCG@10": r.get("ndcg_10", 0),
                        "NDCG@50": r.get("ndcg_50", 0),
                        "MAP": r.get("map", 0),
                        "P@10": r.get("p_10", 0),
                    })
            
            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(df.style.highlight_max(axis=0, subset=["Composite", "NDCG@10", "NDCG@50", "MAP", "P@10"]), use_container_width=True)
                
                # Bar chart comparison
                fig = go.Figure()
                for metric in ["Composite", "NDCG@10", "NDCG@50", "MAP", "P@10"]:
                    fig.add_trace(go.Bar(name=metric, x=df["Method"], y=df[metric]))
                fig.update_layout(
                    barmode="group", title="Ablation Comparison",
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e0e0ff'), height=400,
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Ablation results not yet generated. Run the evaluation harness first.")
        
        # Benchmark results
        st.subheader("⚡ FAISS Scalability Benchmark")
        bench_path = os.path.join(PROJECT_ROOT, "output", "benchmark_results.json")
        if os.path.exists(bench_path):
            with open(bench_path, "r") as f:
                bench = json.load(f)
            
            import pandas as pd
            bench_rows = []
            for key, r in sorted(bench.items(), key=lambda x: x[1]["dataset_size"]):
                bench_rows.append({
                    "Dataset Size": f"{r['dataset_size']:,}",
                    "Index Type": r["index_type"],
                    "Build Time (s)": r["build_time_seconds"],
                    "Search Time (ms)": r["avg_search_time_ms"],
                    "Queries/s": r["searches_per_second"],
                })
            
            if bench_rows:
                st.dataframe(pd.DataFrame(bench_rows), use_container_width=True)
        else:
            st.info("Benchmark not yet run. Execute: `python -m src.evaluation.benchmark`")
    
    # === Tab 4: Job Description ===
    with tab_jd:
        st.subheader("📋 Job Description")
        jd_path = os.path.join(PROJECT_ROOT, "data", "job_description.txt")
        if os.path.exists(jd_path):
            with open(jd_path, "r") as f:
                st.text(f.read())


if __name__ == "__main__":
    main()
