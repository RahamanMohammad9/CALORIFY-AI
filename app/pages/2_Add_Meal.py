import os
import sys
import re
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
from PIL import Image
import pandas as pd
import httpx

from database import (
    add_favorite_food,
    create_table,
    get_favorite_foods,
    get_recent_foods,
    insert_meal,
    remove_favorite_food,
)
from model_inference import heuristic_portion_grams, load_model_bundle, predict_topk
from nutrition_service import OpenFoodFactsUnavailableError, resolve_food
from settings import CONFIDENCE_THRESHOLD_DEFAULT
from utils import FAVICON_PATH, apply_glass_style, render_page_header

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Add Meal",
    page_icon=FAVICON_PATH,
    layout="wide"
)

apply_glass_style(st)
create_table()

# -----------------------------
# Setup paths
# -----------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NUTRITION_PATH = os.path.join(PROJECT_ROOT, "nutrition_data.csv")
# -----------------------------
# File checks
# -----------------------------
missing_files = []

for path in [NUTRITION_PATH]:
    if not os.path.exists(path):
        missing_files.append(path)

if missing_files:
    st.error("Missing required files:")
    for path in missing_files:
        st.write(f"- {path}")
    st.stop()

# -----------------------------
# Cached loaders
# -----------------------------
@st.cache_data
def load_nutrition_data():
    return pd.read_csv(NUTRITION_PATH)


class_names, _model, device = load_model_bundle()
nutrition_df = load_nutrition_data()

# -----------------------------
# Helpers
# -----------------------------
def get_nutrition(food_name):
    row = nutrition_df[nutrition_df["food"] == food_name]

    if not row.empty:
        return {
            "calories_per_100g": float(row["calories_per_100g"].values[0]),
            "protein": float(row["protein"].values[0]),
            "carbs": float(row["carbs"].values[0]),
            "fat": float(row["fat"].values[0]),
            "found": True
        }

    return {
        "calories_per_100g": 250.0,
        "protein": 5.0,
        "carbs": 30.0,
        "fat": 10.0,
        "found": False
    }


def parse_voice_items(text):
    cleaned = str(text).lower().strip()
    # Remove common conversational prefixes to improve parsing.
    cleaned = re.sub(r"^(i\s+(ate|had|have|consumed)\s+)", "", cleaned)
    cleaned = re.sub(r"^(today\s+i\s+(ate|had)\s+)", "", cleaned)
    parts = re.split(r",| and ", cleaned)
    items = []
    for raw in parts:
        token = raw.strip()
        if not token:
            continue
        # Pattern: optional quantity + optional unit + food name
        m = re.match(r"(?:(\d+(?:\.\d+)?)\s*(x|piece|pieces|slice|slices|cup|cups)?\s+)?(.+)", token)
        if not m:
            continue
        qty = float(m.group(1)) if m.group(1) else 1.0
        name = m.group(3).strip()
        name = re.sub(r"^(of\s+)", "", name).strip()
        # Singularize simple plurals (eggs -> egg) for better resolver matching.
        if name.endswith("es") and len(name) > 4:
            name_alt = name[:-2]
        elif name.endswith("s") and len(name) > 3:
            name_alt = name[:-1]
        else:
            name_alt = name
        if name in {"eggs", "egg"}:
            name_alt = "omelette"
        if name in {"toast"}:
            name_alt = "garlic bread"
        if not name:
            continue
        # Simple default portion mapping: 1 item ~ 80g
        grams_est = max(20.0, qty * 80.0)
        items.append((name_alt, qty, grams_est))
    return items


def try_resolve_voice_items(items):
    resolved = []
    unresolved = []
    for name, qty, grams_est in items:
        try:
            probe = resolve_food(name, float(grams_est))
            resolved.append((name, qty, grams_est, probe.source, probe.confidence))
        except Exception:
            unresolved.append((name, qty, grams_est))
    return resolved, unresolved


render_page_header(
    st,
    "Add Meal",
    "Add a meal from a photo (AI) or by typing a food name (local list + Open Food Facts)",
    kicker="Log food",
)

with st.expander("About this prediction"):
    st.write("This AI estimate is a guidance tool, not medical advice.")
    st.write("- The model predicts from known training classes; uncommon dishes may be misclassified.")
    st.write("- Confidence score is not guaranteed correctness.")
    st.write("- Portion and nutrition are estimates; verify labels or use manual edits when needed.")
    st.write("- For low confidence (<60%), always confirm before saving.")

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Meal Settings")

    meal_log_date = st.date_input(
        "Meal date",
        value=date.today(),
        key="addmeal_meal_date",
        help="Which calendar day this meal counts toward (dashboard “today” uses your PC’s date).",
    )

    grams = st.number_input(
        "Portion weight (grams)",
        min_value=1,
        max_value=2000,
        value=100,
        step=10
    )
    confidence_threshold = st.slider(
        "Confidence threshold (%)",
        min_value=40,
        max_value=95,
        value=max(40, min(95, int(CONFIDENCE_THRESHOLD_DEFAULT))),
        step=5,
        help="Predictions below this threshold require explicit confirmation before saving.",
    )
    local_only_lookup = st.checkbox(
        "Local-only nutrition lookup (demo safe)",
        value=False,
        help="When enabled, name lookup uses local nutrition_data.csv only and skips Open Food Facts.",
    )

    st.markdown("---")
    st.write(f"**Model device:** `{device}`")
    st.write(f"**Food classes loaded:** `{len(class_names)}`")
    st.caption("Portion weight below applies to the **From photo** tab.")

# -----------------------------
# Tabs: photo vs name
# -----------------------------
tab_photo, tab_name = st.tabs(["From photo", "By name"])

with tab_photo:
    left_col, right_col = st.columns([1.1, 1])

    with left_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Upload Meal Image")

        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"],
            key="addmeal_photo_uploader",
        )
        detect_multiple = st.checkbox(
            "Detect multiple foods in this image",
            key="addmeal_multi_detect",
            help="Use top predictions as separate meal items with individual portion sliders.",
        )

        selected_food = None
        predicted_food = None
        confidence_value = None
        final_calories = None
        final_protein = None
        final_carbs = None
        final_fat = None
        calories_per_100g = None
        protein_per_100g = None
        carbs_per_100g = None
        fat_per_100g = None

        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert("RGB")
            except Exception:
                st.error("Invalid or corrupted image file. Please upload a valid JPG/PNG image.")
                st.stop()
            st.image(image, caption="Uploaded image", width="stretch")
            top_preds = predict_topk(image, top_k=3, use_tta=True)
            predicted_food = top_preds[0][0]
            confidence_value = top_preds[0][1]

            st.markdown("### Top Predictions")
            for pred_name, pred_conf in top_preds:
                st.progress(min(int(pred_conf), 100), text=f"{pred_name}: {pred_conf:.2f}%")

            is_low_conf = confidence_value < float(confidence_threshold)
            if is_low_conf:
                st.warning(
                    f"Low confidence ({confidence_value:.1f}% < {confidence_threshold}%). "
                    "Please confirm this prediction before saving."
                )
            confirm_low_conf = st.checkbox(
                "I confirm this prediction is correct",
                key="confirm_low_conf",
                value=not is_low_conf,
            )

            if detect_multiple:
                st.markdown("### Multi-food selection")
                top3_names = [n for n, _p in top_preds]
                selected_multi = st.multiselect(
                    "Choose foods present in this image",
                    options=top3_names,
                    default=top3_names[:2],
                    key="addmeal_photo_multi_foods",
                )
                multi_items = []
                for idx, food_name in enumerate(selected_multi):
                    portion = st.slider(
                        f"{food_name} portion (g)",
                        min_value=20,
                        max_value=600,
                        value=120,
                        step=10,
                        key=f"addmeal_photo_multi_portion_{idx}",
                    )
                    nutr = get_nutrition(food_name)
                    item_cal = (nutr["calories_per_100g"] * portion) / 100
                    item_prot = (nutr["protein"] * portion) / 100
                    item_carb = (nutr["carbs"] * portion) / 100
                    item_fat = (nutr["fat"] * portion) / 100
                    multi_items.append((food_name, portion, item_cal, item_prot, item_carb, item_fat))
                    st.caption(
                        f"{food_name}: {item_cal:.1f} kcal | P {item_prot:.1f}g | C {item_carb:.1f}g | F {item_fat:.1f}g"
                    )

                if st.button("💾 Save Selected Meals", width="stretch", key="addmeal_save_multi_photo"):
                    if not multi_items:
                        st.warning("Select at least one food item first.")
                    elif is_low_conf and not confirm_low_conf:
                        st.warning("Please confirm low-confidence prediction before saving.")
                    else:
                        for item in multi_items:
                            insert_meal(
                                food_name=item[0],
                                grams=item[1],
                                calories=item[2],
                                protein=item[3],
                                carbs=item[4],
                                fat=item[5],
                                confidence=confidence_value,
                                meal_date=meal_log_date,
                            )
                        st.success(f"Saved {len(multi_items)} meal item(s) successfully.")
            else:
                st.markdown("### Confirm or Change Food Item")
                default_index = class_names.index(predicted_food) if predicted_food in class_names else 0

                selected_food = st.selectbox(
                    "Select the correct food item",
                    options=class_names,
                    index=default_index,
                    key="addmeal_photo_select_food",
                )

                nutrition = get_nutrition(selected_food)

                calories_per_100g = nutrition["calories_per_100g"]
                protein_per_100g = nutrition["protein"]
                carbs_per_100g = nutrition["carbs"]
                fat_per_100g = nutrition["fat"]

                default_est_grams = int(round(heuristic_portion_grams(confidence_value, calories_per_100g)))
                est_grams = st.slider(
                    "AI estimated portion (g)",
                    min_value=40,
                    max_value=400,
                    value=max(40, min(400, default_est_grams)),
                    step=10,
                    key="addmeal_portion_estimate",
                    help="Heuristic estimate; adjust as needed.",
                )
                portion_grams = float(est_grams if est_grams > 0 else grams)

                final_calories = (calories_per_100g * portion_grams) / 100
                final_protein = (protein_per_100g * portion_grams) / 100
                final_carbs = (carbs_per_100g * portion_grams) / 100
                final_fat = (fat_per_100g * portion_grams) / 100

                if not nutrition["found"]:
                    try:
                        resolved = resolve_food(selected_food, portion_grams)
                        final_calories = float(resolved.calories)
                        final_protein = float(resolved.protein)
                        final_carbs = float(resolved.carbs)
                        final_fat = float(resolved.fat)
                        st.info("Nutrition estimated from fallback lookup (Open Food Facts/local resolver).")
                    except Exception:
                        st.warning("Nutrition data not available for this food. Using safe fallback values.")

                if st.button("💾 Save Meal", width="stretch", key="addmeal_save_photo"):
                    if is_low_conf and not confirm_low_conf:
                        st.warning("Please confirm low-confidence prediction before saving.")
                    else:
                        insert_meal(
                            food_name=selected_food,
                            grams=portion_grams,
                            calories=final_calories,
                            protein=final_protein,
                            carbs=final_carbs,
                            fat=final_fat,
                            confidence=confidence_value,
                            meal_date=meal_log_date,
                        )
                        st.success("Meal saved successfully.")
        else:
            st.info("Upload an image to start.")
        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Meal Summary")

        if uploaded_file is None:
            st.write("Upload an image to view prediction and nutrition details.")
        else:
            st.write(f"**Model Prediction:** {predicted_food}")
            if not detect_multiple:
                st.write(f"**Selected Food:** {selected_food}")
            st.write(f"**Confidence:** {confidence_value:.2f}%")
            st.write(f"**Portion Weight (configured):** {grams} g")

            if not detect_multiple:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Calories", f"{final_calories:.1f} kcal")
                m2.metric("Protein", f"{final_protein:.1f} g")
                m3.metric("Carbs", f"{final_carbs:.1f} g")
                m4.metric("Fat", f"{final_fat:.1f} g")

                st.markdown("### Reference Values per 100g")
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Calories/100g", f"{calories_per_100g:.1f}")
                r2.metric("Protein/100g", f"{protein_per_100g:.1f}")
                r3.metric("Carbs/100g", f"{carbs_per_100g:.1f}")
                r4.metric("Fat/100g", f"{fat_per_100g:.1f}")
            else:
                st.info("Multi-food mode enabled. Review and save from the left panel.")
        st.markdown('</div>', unsafe_allow_html=True)

with tab_name:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Add by food name")
    st.caption(
        "Look up from your dish list or Open Food Facts, then correct the numbers if needed—or enter everything yourself."
    )

    manual_entry = st.checkbox(
        "Enter nutrition manually (skip lookup)",
        key="addmeal_manual_entry",
        help="Use when lookup fails or you already know calories and macros for this portion.",
    )

    name_food = st.text_input(
        "Food name",
        placeholder='e.g. "caesar salad", "chocolate bar"',
        key="addmeal_name_food",
        help="Shown in your meal history. For manual entry this can be any label you want.",
    )
    name_grams = st.number_input(
        "Portion (grams)",
        min_value=1.0,
        max_value=10000.0,
        value=100.0,
        step=10.0,
        key="addmeal_name_grams",
    )

    name_lookup_key = (name_food.strip().lower(), float(name_grams))

    def _clear_name_lookup_state():
        st.session_state.pop("addmeal_name_result", None)
        st.session_state.pop("addmeal_name_key", None)
        for _k in (
            "addmeal_portion_cal",
            "addmeal_portion_prot",
            "addmeal_portion_carb",
            "addmeal_portion_fat",
        ):
            st.session_state.pop(_k, None)

    if manual_entry:
        st.markdown("##### Nutrition for this portion")
        st.caption("Values below are saved as-is (for this portion size).")
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            man_cal = st.number_input(
                "Calories (kcal)",
                min_value=0.0,
                max_value=50000.0,
                step=1.0,
                value=0.0,
                key="addmeal_manual_cal",
            )
        with mc2:
            man_prot = st.number_input(
                "Protein (g)",
                min_value=0.0,
                max_value=5000.0,
                step=0.5,
                value=0.0,
                key="addmeal_manual_prot",
            )
        with mc3:
            man_carb = st.number_input(
                "Carbs (g)",
                min_value=0.0,
                max_value=5000.0,
                step=0.5,
                value=0.0,
                key="addmeal_manual_carb",
            )
        with mc4:
            man_fat = st.number_input(
                "Fat (g)",
                min_value=0.0,
                max_value=5000.0,
                step=0.5,
                value=0.0,
                key="addmeal_manual_fat",
            )

        if st.button("Save to tracker", width="stretch", type="primary", key="addmeal_save_manual"):
            if not name_food.strip():
                st.warning("Enter a food name.")
            elif man_cal <= 0:
                st.warning("Enter calories for this portion (greater than 0).")
            else:
                meal_id = insert_meal(
                    food_name=name_food.strip(),
                    grams=float(name_grams),
                    calories=float(man_cal),
                    protein=float(man_prot),
                    carbs=float(man_carb),
                    fat=float(man_fat),
                    confidence=100.0,
                    meal_date=meal_log_date,
                )
                st.success(f"Saved manually as meal **#{meal_id}**.")
    else:
        nc1, nc2 = st.columns(2)
        with nc1:
            name_lookup = st.button("Look up nutrition", width="stretch", key="addmeal_name_lookup")
        with nc2:
            name_save = st.button("Save to tracker", width="stretch", type="primary", key="addmeal_name_save")

        if name_lookup:
            if not name_food.strip():
                st.warning("Enter a food name.")
            else:
                try:
                    if local_only_lookup:
                        local_row = nutrition_df[nutrition_df["food"] == name_food.strip().lower().replace(" ", "_")]
                        if local_row.empty:
                            raise LookupError(
                                "No local match found. Disable local-only mode or use a food from the local dataset."
                            )
                        cph = float(local_row["calories_per_100g"].values[0])
                        pph = float(local_row["protein"].values[0])
                        carbph = float(local_row["carbs"].values[0])
                        fph = float(local_row["fat"].values[0])
                        factor = float(name_grams) / 100.0
                        class _LocalResolved:
                            def __init__(self):
                                self.display_name = name_food.strip().title()
                                self.matched_key = name_food.strip().lower().replace(" ", "_")
                                self.calories = cph * factor
                                self.protein = pph * factor
                                self.carbs = carbph * factor
                                self.fat = fph * factor
                                self.calories_per_100g = cph
                                self.protein_per_100g = pph
                                self.carbs_per_100g = carbph
                                self.fat_per_100g = fph
                                self.source = "local_csv"
                                self.confidence = 1.0
                        resolved = _LocalResolved()
                    else:
                        resolved = resolve_food(name_food.strip(), float(name_grams))
                    st.session_state["addmeal_name_result"] = resolved
                    st.session_state["addmeal_name_key"] = name_lookup_key
                    st.session_state["addmeal_portion_cal"] = float(round(resolved.calories, 2))
                    st.session_state["addmeal_portion_prot"] = float(round(resolved.protein, 2))
                    st.session_state["addmeal_portion_carb"] = float(round(resolved.carbs, 2))
                    st.session_state["addmeal_portion_fat"] = float(round(resolved.fat, 2))
                except ValueError as e:
                    st.error(str(e))
                    _clear_name_lookup_state()
                except LookupError as e:
                    st.error(str(e))
                    _clear_name_lookup_state()
                except OpenFoodFactsUnavailableError as e:
                    st.error(str(e))
                    _clear_name_lookup_state()
                except httpx.HTTPError as e:
                    st.error(f"Network error: {e}")
                    _clear_name_lookup_state()

        stored = st.session_state.get("addmeal_name_result")
        stored_key = st.session_state.get("addmeal_name_key")

        if stored is not None and stored_key == name_lookup_key and name_food.strip():
            src = "Local dish list" if stored.source == "local_csv" else "Open Food Facts"
            st.caption(
                f"Source: **{src}** · match: `{stored.matched_key}` · confidence ~{stored.confidence * 100:.0f}%"
            )
            st.markdown("##### Values for this portion (edit if wrong)")
            ec1, ec2, ec3, ec4 = st.columns(4)
            with ec1:
                st.number_input(
                    "Calories (kcal)",
                    min_value=0.0,
                    max_value=50000.0,
                    step=1.0,
                    key="addmeal_portion_cal",
                )
            with ec2:
                st.number_input(
                    "Protein (g)",
                    min_value=0.0,
                    max_value=5000.0,
                    step=0.5,
                    key="addmeal_portion_prot",
                )
            with ec3:
                st.number_input(
                    "Carbs (g)",
                    min_value=0.0,
                    max_value=5000.0,
                    step=0.5,
                    key="addmeal_portion_carb",
                )
            with ec4:
                st.number_input(
                    "Fat (g)",
                    min_value=0.0,
                    max_value=5000.0,
                    step=0.5,
                    key="addmeal_portion_fat",
                )
            with st.expander("Per 100 g reference (from lookup)"):
                st.write(
                    f"Calories {stored.calories_per_100g:.1f} kcal · "
                    f"P {stored.protein_per_100g:.1f} g · "
                    f"C {stored.carbs_per_100g:.1f} g · "
                    f"F {stored.fat_per_100g:.1f} g"
                )

        if name_save:
            s = st.session_state.get("addmeal_name_result")
            sk = st.session_state.get("addmeal_name_key")
            if not name_food.strip():
                st.warning("Enter a food name.")
            elif s is None or sk != name_lookup_key:
                st.warning(
                    "Click **Look up nutrition** first, or enable **Enter nutrition manually** above."
                )
            else:
                cal_v = float(st.session_state.get("addmeal_portion_cal", s.calories))
                p_v = float(st.session_state.get("addmeal_portion_prot", s.protein))
                c_v = float(st.session_state.get("addmeal_portion_carb", s.carbs))
                f_v = float(st.session_state.get("addmeal_portion_fat", s.fat))
                if cal_v <= 0:
                    st.warning("Calories must be greater than 0.")
                else:
                    macros_edited = (
                        abs(cal_v - s.calories) > 0.51
                        or abs(p_v - s.protein) > 0.51
                        or abs(c_v - s.carbs) > 0.51
                        or abs(f_v - s.fat) > 0.51
                    )
                    save_conf = 92.0 if macros_edited else s.confidence * 100.0
                    meal_id = insert_meal(
                        food_name=s.display_name,
                        grams=float(name_grams),
                        calories=cal_v,
                        protein=p_v,
                        carbs=c_v,
                        fat=f_v,
                        confidence=save_conf,
                        meal_date=meal_log_date,
                    )
                    st.success(f"Saved as meal **#{meal_id}**.")
                    _clear_name_lookup_state()

        if (
            not manual_entry
            and st.session_state.get("addmeal_name_result") is not None
            and st.session_state.get("addmeal_name_key") != name_lookup_key
        ):
            st.info("Name or portion changed — click **Look up nutrition** again before saving.")

    st.caption("Open Food Facts estimates are not medical advice.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Quick Add: Recent and Favorites")

recent_foods = get_recent_foods(limit=8)
favorite_foods = get_favorite_foods()
quick_food_options = favorite_foods + [f for f in recent_foods if f not in favorite_foods]

if quick_food_options:
    q1, q2 = st.columns([1.4, 1])
    with q1:
        quick_food = st.selectbox("Pick a recent/favorite food", options=quick_food_options, key="quick_add_food")
    with q2:
        quick_grams = st.number_input(
            "Portion (g)",
            min_value=20.0,
            max_value=2000.0,
            value=100.0,
            step=10.0,
            key="quick_add_grams",
        )

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("Save quick meal", width="stretch", key="quick_add_save"):
            try:
                resolved = resolve_food(quick_food, float(quick_grams))
                insert_meal(
                    food_name=resolved.display_name,
                    grams=float(quick_grams),
                    calories=float(resolved.calories),
                    protein=float(resolved.protein),
                    carbs=float(resolved.carbs),
                    fat=float(resolved.fat),
                    confidence=float(resolved.confidence * 100.0),
                    meal_date=meal_log_date,
                )
                st.success("Quick meal saved.")
            except Exception as e:
                st.error(f"Could not save quick meal: {e}")
    with b2:
        if st.button("Add to favorites", width="stretch", key="quick_add_fav"):
            add_favorite_food(quick_food)
            st.success("Added to favorites.")
    with b3:
        if st.button("Remove favorite", width="stretch", key="quick_add_unfav"):
            remove_favorite_food(quick_food)
            st.info("Removed from favorites.")
else:
    st.info("No recent/favorite foods yet. Save a few meals first.")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Voice-style Logging")
voice_text = st.text_input(
    "Type a natural meal sentence",
    placeholder='e.g. "I ate 2 eggs and toast"',
    key="voice_like_input",
)
if voice_text.strip():
    parsed_items = parse_voice_items(voice_text)
    if parsed_items:
        resolved_preview, unresolved_preview = try_resolve_voice_items(parsed_items)
        st.caption("Parsed items (editable via estimated grams):")
        if resolved_preview:
            st.markdown("##### Resolution preview")
            for name, qty, grams_est, src, conf in resolved_preview:
                source_name = "Local CSV" if src == "local_csv" else "Open Food Facts"
                st.write(f"- **{name}** ({qty:g}x) ~{grams_est:.0f}g · source: {source_name} · confidence: {conf*100:.0f}%")
        if unresolved_preview:
            st.warning("Some items could not be resolved. Please rewrite these names before saving:")
            for name, qty, _g in unresolved_preview:
                st.write(f"- `{name}` ({qty:g}x)")

        edits = []
        for i, (name, qty, grams_est) in enumerate(parsed_items):
            g = st.number_input(
                f"{name} ({qty:g}x) estimated grams",
                min_value=20.0,
                max_value=1000.0,
                value=float(grams_est),
                step=10.0,
                key=f"voice_item_grams_{i}",
            )
            edits.append((name, g))
        if st.button("Save parsed meal items", width="stretch", key="voice_like_save"):
            saved = 0
            failed = []
            for name, grams_v in edits:
                try:
                    resolved = resolve_food(name, float(grams_v))
                    insert_meal(
                        food_name=resolved.display_name,
                        grams=float(grams_v),
                        calories=float(resolved.calories),
                        protein=float(resolved.protein),
                        carbs=float(resolved.carbs),
                        fat=float(resolved.fat),
                        confidence=float(resolved.confidence * 100.0),
                        meal_date=meal_log_date,
                    )
                    saved += 1
                except Exception:
                    failed.append(name)
            if saved > 0:
                st.success(f"Saved {saved} parsed item(s).")
            if failed:
                st.warning("Could not resolve: " + ", ".join(sorted(set(failed))) + ". Try simpler/common food names.")
            if saved == 0 and failed:
                st.error("No items were saved.")
    else:
        st.info("Could not parse items from this sentence yet. Try comma-separated foods.")
st.markdown("</div>", unsafe_allow_html=True)