# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The main app."""

import data_processing
from langchain_core.prompts import FewShotPromptTemplate
from langchain_core.prompts import PromptTemplate
import llm_services
import pandas as pd
import rag_utils
import streamlit as st
import ui
import utils

# Page setup.
st.set_page_config(
    page_title="Description Genius", page_icon=":owl:", layout="wide"
)
st.title("Description Genius 🦉")
st.session_state.operation_mode = st.segmented_control(
    "Select operation mode",
    [
        "Generate Descriptions",
        "Translate existing descriptions",
    ],
    default="Generate Descriptions",
    key="operation_mode_selector",
)

# Render UI.
gcp_id, region, llm_model_name, temperature = ui.render_sidebar()
image_column = None
input_df = pd.DataFrame()

if st.session_state.operation_mode == "Generate Descriptions":
  # Data upload and clean up functionality.
  with st.expander("**Data Upload**", expanded=True):
    input_data = st.file_uploader("Upload your CSV table", type=["csv"])
    prompt_features_list = []
    prompt_features = []
    use_image_for_generation = False
    if input_data:
      input_df = data_processing.load_dataframe(input_data)
      image_column = st.selectbox(
          "Image Column",
          input_df.columns.tolist(),
          index=None,
          placeholder="Select an image column if available.",
      )
      use_image_for_generation = st.checkbox(
          "Use image attributes for description",
          value=False,
          disabled=image_column is None,
          help="Extract additional product attributes from the given image.",
      )
      input_data_columns_config = {}
      if image_column is not None:
        input_data_columns_config[image_column] = st.column_config.ImageColumn()
      remove_html_tags = st.checkbox("Remove html tags", value=True)
      processed_df = data_processing.preprocess_dataframe(
          input_df, remove_html_tags
      )
      ui.render_dataframe(processed_df, input_data_columns_config)
      with st.container():
        input_columns = input_df.columns.to_list()
        columns_for_prompt = st.multiselect(
            "Columns to use in prompt",
            input_columns,
            input_columns,
            placeholder="Select columns...",
        )
        remove_empty_values = st.checkbox(
            "Ignore empty or NaN values", value=True
        )
      prompt_df = processed_df[columns_for_prompt].copy()
      prompt_features_str = prompt_df.astype(str).apply(
          utils.row_to_custom_str, args=(remove_empty_values,), axis=1
      )
      prompt_features_list = prompt_features_str.to_list()
      if image_column is not None and use_image_for_generation:
        image_links_list = processed_df[image_column].to_list()
        prompt_features = [{
            "input_features": val,
            "image_url": url
        } for val, url in zip(prompt_features_list, image_links_list)]
      else:
        prompt_features = [{
            "input_features": val
        } for val in prompt_features_list]

  # Generation form and settings.
  with st.form("generation_config"):
    st.write("**Prompt**")
    prompt_llm_role = st.text_input(
        "Provide a role to the LLM",
        value=(
            "You are the Lead E-commerce Copywriter for Ferns N Petals (FNP). Your goal is to write clear, "
            "scannable, and informative product descriptions that balance light emotional resonance with strict "
            "factual clarity. The customer must know exactly what the product is, its size/specifications, and its "
            "presentation within the first glance."
        ),
    )
    prompt_llm_guidelines = st.text_area(
        "Provide any guidelines that the LLM should consider",
        height=150,
        value=(
            "###GUIDELINES###\n"
            "1. IMMEDIATE PRODUCT IDENTIFICATION (SEO): You must explicitly mention the exact Product Name in the first sentence. "
            "Never hide the product behind vague poetic descriptors (e.g., do not substitute \"Chocolate Truffle Cake\" with \"decadent dessert\").\n"
            "2. VERSATILE GIFTING CONTEXTS: Do not pigeonhole products into narrow, highly specific scenarios. Keep the gifting intent broad, "
            "natural, and highly inclusive of various milestones like birthdays, anniversaries, expressions of gratitude, or celebrations.\n"
            "3. HARD MARKETING-FLUFF BAN: Completely eliminate abstract, over-the-top marketing clichés. Zero tolerance for: \"ignite romance,\" "
            "\"intimate candlelit dinners,\" \"shared moments of pure joy,\" \"sophisticated palates,\" \"personal sanctuary,\" \"understated luxury,\" "
            "\"breathtaking gesture,\" or \"perfect surprise.\"\n"
            "4. GRAMMATICAL PRECISION: Ensure absolute flawless grammar throughout. Pay strict attention to indefinite articles before adjectives "
            "(e.g., always use \"An elegant presentation box\" instead of \"A elegant presentation box\").\n\n"
            "To ensure sequential products do not read identically, vary your sentence transitions and completely diversify the final sentence focus.\n"
            "- COMPOSITION: Output must be a SINGLE, continuous paragraph.\n"
            "- FORMAT: Clean, raw plain text only. No markdown asterisks (**) or HTML tags.\n"
            "- LENGTH: Exactly 3 sentences. Hard maximum of 50 words total.\n\n"
            "###THREE-SENTENCE FUNCTIONAL BLUEPRINT###\n"
            "* SENTENCE 1 (The Clear Opening): State the exact Product Name immediately and pair it with a natural, versatile milestone or celebration context. (Do NOT start with \"Welcome to\" or \"Celebrate life's\").\n"
            "* SENTENCE 2 (The Factual Breakdown): Detail the exact specifications (weights, counts, ingredients, or sizes using clear numbers like \"6\" or \"500g\") in a crisp narrative. Never use the word \"features\", \"contains\", or \"includes\".\n"
            "* SENTENCE 3 (The Diverse Presentation Close): Complete the description by highlighting the premium packaging, the sensory freshness, or the immediate display aesthetic.\n"
            "* SENTENCE 3 CRITICAL BAN: You are strictly forbidden from using the words \"arrives\", \"delivered\", \"delivery\", \"receive\", or \"recipient\" in this final sentence. Focus entirely on the object's presentation, structural protection, or visual beauty."
        ),
    )

    examples_df = pd.DataFrame([{
        "input": "Product Name: Chocolate Truffle Cake Half Kg, Product Details: 500g, eggless, round shape, chocolate ganache glaze, chocolate swirls decoration",
        "output": (
            "Make any birthday or milestone celebration memorable with our Chocolate Truffle Cake Half Kg. "
            "This round, 500g eggless cake is crafted with a smooth chocolate ganache glaze and topped with handcrafted chocolate swirls. "
            "A glossy finish and elegant piping ensure this dessert stands out as a stunning centerpiece on any party table."
        ),
    }])
    prompt_action = st.text_input(
        "Generation prompt",
        value=(
            "Generate text descriptions based on the given ###FEATURES### and"
            " ###GUIDELINES###."
        ),
    )
    st.write("**Few-Shot examples**")
    edited_df = st.data_editor(
        examples_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
    )
    example_selection_criteria = st.radio(
        label="Example selection similarity",
        options=["min", "max"],
        captions=[
            (
                "Select the example which is least similar to our input"
                " (prevents over-fitting)"
            ),
            (
                "Select the example which is most similar to our input"
                " (prevents hallucination)"
            ),
        ],
    )

    prompt_additional_context = None  # Variable so pylint: disable=C0103

    _PROMPT_SUFFIX = """
        Input features: {input_features}
        """

    with st.expander("Advanced"):
      additional_context_col, forbidden_words_col = st.columns(2)
      additional_context_col.write("**Additional Context**")
      context_docs = additional_context_col.file_uploader(
          "Upload context documents", type=["txt"], accept_multiple_files=True
      )
      if context_docs:
        # Use retrieval augmented generation (RAG) to fetch information.
        context_list = rag_utils.get_context_list_from_docs(
            context_docs, prompt_features_list
        )
        # Add additional context to the prompt_features dictionary.
        for feature, context in zip(prompt_features, context_list):
          feature["additional_context"] = context
        prompt_additional_context = (
            "\nNow I will provide some Additional Context for generating the"
            " descriptions: {additional_context}\n\n"
        )

      forbidden_words_col.write("**Forbidden Words**")
      filter_words_str = None
      filter_words_str = forbidden_words_col.text_area(
          "List of forbidden words",
          height=100,
          placeholder=(
              "Enter a comma-separated list of words that should not occur in"
              " the results."
          ),
      )
      enable_word_filtering = forbidden_words_col.toggle(
          "Enable forbidden word scanning"
      )

      st.write("**Scoring**")
      scoring_template = None  # Variable so pylint: disable=C0103
      default_scoring_data = [
          {
              "Criterion": "CMS Layout & Length Compliance: Must be a single continuous paragraph of exactly 3 sentences. Hard maximum of 60 words total. Score 0/30 if markdown asterisks (**) or HTML tags are present.",
              "Points": 30
          },
          {
              "Criterion": "Brand Voice Safety: Confirm zero presence of these specific banned cliches: 'breathtaking gesture of love,' 'ethereal,' 'symphony of flavors,' or 'magical aura.'",
              "Points": 30
          },
          {
              "Criterion": "Zero-Hallucination SKU Accuracy: Ensure the copy naturally includes the exact Product Name in the first sentence for SEO mapping, and that the physical specifications (like grams, counts, or sizes) match logically.",
              "Points": 25
          },
          {
              "Criterion": "Gifting Focus & Presentation Close: Gifting Intent Rule: Accept either a single specific occasion OR a versatile mix of milestone celebrations (e.g., birthdays, anniversaries, housewarmings). Do not penalize versatile gifting options. Presentation Close Rule: The final sentence must highlight how the product is presented (physical packaging like boxes, wrapping, ribbons, bows OR premium product display elements like elegant piping, glossy finishes, decorative river pebbles). If met, award full points (15/15) and output Passing: Y.",
              "Points": 15
          }
      ]
      scoring_df = pd.DataFrame(default_scoring_data)
      scoring_criteria = st.data_editor(
          data=scoring_df,
          column_config={
              "Criterion": st.column_config.TextColumn(width="large"),
              "Points": st.column_config.NumberColumn(
                  "Points",
                  help=(
                      "Assign a numeric value to signify the importance of a"
                      " particular criterion in the total scoring e.g."
                      " **Criterion Points > Min. passing score** ensure the"
                      " entire description fails the quality check if this"
                      " criterion is not passed."
                  ),
                  width="small",
                  required=True,
              ),
          },
          num_rows="dynamic",
          use_container_width=True,
          hide_index=True,
      )
      scoring_prompt_lines = []
      if not scoring_criteria.empty:
          for _, row in scoring_criteria.iterrows():
              if pd.notna(row['Criterion']) and pd.notna(row['Points']):
                  scoring_prompt_lines.append(f"- {row['Criterion']} | Weight: {row['Points']} Points")
      scoring_prompt = "\n".join(scoring_prompt_lines)
        
      scoring_total_points_col, passing_score_col = st.columns(2)
      scoring_total_points = scoring_total_points_col.number_input(
          "Total available points",
          0,
          1000,
          value=0,
          help=(
              "Highest score a description can achieve. All descriptions start"
              " with this score and lose points if they fail a criterion."
          ),
      )
      passing_score = passing_score_col.number_input(
          "Minimum score required to pass",
          -1000,
          1000,
          value=0,
          help=(
              "Minimum score required for a description to pass a quality"
              " check."
          ),
      )
      enable_scoring = st.toggle("Enable scoring")

      SCORING_PROMPT_PREFIX = """
                You are a critic with an IQ of 140 and an expert in content creation who scores a generated product description based on the following criteria and points per criterion.

                Instructions for you:
                Read the product description carefully and compare it to the given product attributes.
                Review the quality of the generated description based on the given scoring criteria and output your findings.
                Format it as `Quality Review: ____`. Then, provide a final score based on your quality review. Format it as `Final score is: |SCORE|`.
                Let's work this out step by step.
            """

      SCORING_PROMPT_SUFFIX = """
                Here are the product attributes and the generated product description.
                Do the quality review and provide your score.

                Product Attributes: {input_features}
                Product Description: {generated_description}
                Your quality review and final score:
            """

      scoring_template = PromptTemplate(
          input_variables=["generated_description", "input_features"],
          template=f"{SCORING_PROMPT_PREFIX}\n{scoring_prompt}\n{SCORING_PROMPT_SUFFIX}",
      )

    prompt_input_variables = ["input_features"]
    if prompt_additional_context:
      prompt_input_variables.append("additional_context")

    prompt_prefix = f"{prompt_llm_role}\n{prompt_llm_guidelines}\n{prompt_additional_context}\n{prompt_action}"

    # If examples have been provided, use a Few Shot Prompt.
    edited_df.dropna(how="all", inplace=True)
    if not edited_df.empty:
      example_prompt = PromptTemplate(
          input_variables=["input", "output"],
          template="Input features: {input}\nOutput description: {output}",
      )
      example_selector = rag_utils.CustomSimilarityExampleSelector(
          examples=edited_df.to_dict("records"),
          ex_prompt=example_prompt,
          selection_criteria=example_selection_criteria,
          k=2,
      )

      description_template = FewShotPromptTemplate(
          example_selector=example_selector,
          example_prompt=example_prompt,
          prefix=prompt_prefix,
          suffix=_PROMPT_SUFFIX,
          input_variables=prompt_input_variables,
      )
    # If no examples provided, use a standard prompt instead.
    else:
      description_template = PromptTemplate.from_template(
          f"{prompt_prefix}\n{_PROMPT_SUFFIX}"
      )

    generate_button = st.form_submit_button(
        "Generate",
        disabled=not (gcp_id and input_data),
        type="primary",
    )

  if "results" not in st.session_state:
    st.session_state.results = []

  # Results Display and Generation Logic.
  results_placeholder = st.empty()

  if generate_button:
    # Clear previous results for a new generation run
    st.session_state.results = []
    results_placeholder.empty()

    with st.spinner("Generating descriptions..."):
      generated_count = 0
      total_to_generate = len(prompt_features)
      progress_bar = st.progress(
          0, text=f"Generating... (0/{total_to_generate})"
      )

      intermediate_generated_results = []
      for i, result in enumerate(
          llm_services.fetch_response(
              gcp_id,
              region,
              description_template,
              prompt_features,
              llm_model_name,
              temperature,
              prompt_additional_context is not None,
              use_image_for_generation and image_column is not None,
          )
      ):
        intermediate_generated_results.append(result)
        # Append to session_state for progressive display of un-scored results
        st.session_state.results.append(result)

        generated_count += 1
        progress_text = f"Generating... ({generated_count}/{total_to_generate})"
        progress_bar.progress(
            generated_count / total_to_generate, text=progress_text
        )
        ui.display_progressive_results(
            st.session_state.results, results_placeholder
        )
      progress_bar.empty()

    if intermediate_generated_results:  # Check if anything was generated
      if enable_scoring:
        with st.spinner("Scoring descriptions..."):
          results_evals = llm_services.score_descriptions(
              gcp_id,
              region,
              llm_model_name,
              intermediate_generated_results,  # Score the items.
              scoring_criteria.to_dict("records"),
              scoring_total_points,
              passing_score,
          )
          if results_evals:
            st.session_state.results = results_evals
            ui.display_progressive_results(
                st.session_state.results, results_placeholder
            )  # Show scored results in temporary table.
          else:  # Scoring failed or returned empty
            # st.session_state.results still holds the un-scored items
            st.warning(
                "An error occurred when scoring descriptions. Displaying"
                " un-scored results.",
                icon="⚠️",
            )
    else:
      # generate_button was pressed, but intermediate_generated_results empty.
      st.warning("No results were returned from generation.", icon="⚠️")

  # This block will now display the final state of st.session_state.results
  # after generation and optional scoring are complete.
  if st.session_state.results:
    results_df = pd.DataFrame.from_records(data=st.session_state.results)
    results_placeholder.empty()  # Get rid of the temporary table.
    if enable_word_filtering:
      if filter_words_str:
        filter_words = filter_words_str.split(",")
        filter_words = [word.strip() for word in filter_words]
        results_df["contains_forbidden_words"] = results_df[
            "generated_description"].str.contains(
                "|".join(filter_words), na=False
            )
    output_df_column_config = {
        "Select": st.column_config.CheckboxColumn(required=True)
    }
    if (
        image_column is not None and image_column in input_df.columns
        and not results_df.empty
    ):
      num_results_to_display = len(results_df)
      if num_results_to_display <= len(input_df):
        image_values_to_insert = (
            input_df[image_column].iloc[:num_results_to_display].copy()
        )
        results_df.insert(
            loc=0, column=image_column, value=image_values_to_insert
        )
      output_df_column_config[image_column] = st.column_config.ImageColumn()

    # Workaround to render the results DataFrame with selectable rows.
    # TODO(developer): Update this when built-in selection functionality becomes
    # available in Streamlit.
    selected_results = ui.selectable_dataframe(
        results_df, output_df_column_config
    )
    if not selected_results.empty:
      selected_indices = selected_results.index.tolist()
      if "prompt_features" in globals() and all(
          idx < len(prompt_features) for idx in selected_indices
      ):
        regenerate_features = [prompt_features[i] for i in selected_indices]
      else:
        regenerate_features = []
        st.warning(
            "Could not prepare features for regeneration. Indexing issue or"
            " prompt_features not available."
        )

      regenerate_button = st.button("Regenerate Selected")
      if regenerate_button and regenerate_features:
        with st.spinner("Regenerating selected descriptions..."):
          temp_regenerated_items = []
          num_to_regenerate = len(regenerate_features)
          regen_progress_bar = st.progress(
              0, text=f"Regenerating... (0/{num_to_regenerate})"
          )

          for i, result in enumerate(
              llm_services.fetch_response(
                  gcp_id,
                  region,
                  description_template,
                  regenerate_features,
                  llm_model_name,
                  temperature,
                  prompt_additional_context is not None,
                  use_image_for_generation and image_column is not None,
              )
          ):
            temp_regenerated_items.append(result)
            regen_progress_bar.progress(
                (i+1) / num_to_regenerate,
                text=f"Regenerating... ({i+1}/{num_to_regenerate})",
            )
          regen_progress_bar.empty()

          if temp_regenerated_items:
            final_regenerated_results = temp_regenerated_items
            if enable_scoring:
              final_regenerated_results = llm_services.score_descriptions(
                  gcp_id,
                  region,
                  llm_model_name,
                  temp_regenerated_items,
                  scoring_criteria.to_dict("records"),
                  scoring_total_points,
                  passing_score,
              )
            if len(final_regenerated_results) == len(selected_indices):
              for i, original_list_index in enumerate(selected_indices):
                st.session_state.results[original_list_index] = (
                    final_regenerated_results[i]
                )
            else:
              st.warning(
                  "Mismatch in regenerated items count. Update aborted.",
                  icon="⚠️",
              )
            st.rerun()
          else:
            st.warning("No results were returned for regeneration.", icon="⚠️")

    # Download button should use the final results_df
    if not results_df.empty:
      csv = data_processing.convert_df(results_df)
      st.download_button(
          label="Download data as CSV",
          data=csv,
          file_name="text_descriptions.csv",
          mime="text/csv",
      )
elif st.session_state.operation_mode == "Translate existing descriptions":
  source_description_column = None
  with st.expander("**Translate descriptions**", expanded=True):
    input_data = st.file_uploader("Upload your CSV table", type=["csv"])
    if input_data:
      input_df = data_processing.load_dataframe(input_data)
      source_description_column = st.selectbox(
          "Select the column containing the descriptions to translate",
          input_df.columns if input_df is not None else [],
      )
      ui.render_dataframe(input_df)
    st.subheader("Translation Settings")
    target_languages = st.multiselect(
        "Select target languages for translation",
        [
            "Arabic",
            "Bengali",
            "Bulgarian",
            "Chinese",
            "Croatian",
            "Czech",
            "Danish",
            "Dutch",
            "English",
            "Estonian",
            "Finnish",
            "French",
            "German",
            "Greek",
            "Hebrew",
            "Hindi",
            "Hungarian",
            "Indonesian",
            "Italian",
            "Japanese",
            "Korean",
            "Latvian",
            "Lithuanian",
            "Norwegian",
            "Polish",
            "Portuguese",
            "Romanian",
            "Russian",
            "Serbian",
            "Slovak",
            "Slovenian",
            "Spanish",
            "Swahili",
            "Swedish",
            "Thai",
            "Turkish",
            "Ukrainian",
            "Vietnamese",
        ],
    )
    translation_guidelines = st.text_area(
        "Provide guidelines for the translation (e.g., tone, formality per"
        " language)",
        height=150,
        placeholder=(
            "For German, use informal 'du'. For Japanese, maintain a polite"
            " tone."
        ),
    )
    translate_button = st.button(
        "Translate Descriptions from CSV",
        type="primary",
        disabled=not (
            gcp_id and input_data and source_description_column
            and target_languages
        ),
        help=(
            "Make sure you enter all required fields including Cloud"
            " Project ID."
        ),
    )

    if "translation_results" not in st.session_state:
      st.session_state.translation_results = []

    translation_results_placeholder = st.empty()

    if translate_button:
      st.session_state.translation_results = []
      translation_results_placeholder.empty()

      if input_df.empty:
        st.error("The uploaded CSV file is empty.")
      else:
        with st.spinner("Translating descriptions..."):
          # We are only interested in source description column.
          translation_description_df = input_df[source_description_column]
          translated_count = 0
          total_to_translate = len(translation_description_df)
          progress_bar = st.progress(
              0, text=f"Translating... (0/{total_to_translate})"
          )

          intermediate_translated_results = []
          for i, result in enumerate(
              llm_services.translate_texts_to_json_multiple_languages(
                  google_cloud_project_id=gcp_id,
                  region=region,
                  llm_model=llm_model_name,
                  texts_to_translate=translation_description_df.to_list(),
                  target_languages=target_languages,
                  translation_guidelines=translation_guidelines,
                  temperature=temperature,
              )
          ):
            intermediate_translated_results.append(result)
            st.session_state.translation_results.append(result)

            translated_count += 1
            progress_text = (
                f"Translating... ({translated_count}/{total_to_translate})"
            )
            progress_bar.progress(
                translated_count / total_to_translate, text=progress_text
            )
            ui.display_progressive_results(
                st.session_state.translation_results,
                translation_results_placeholder,
            )
          progress_bar.empty()
else:
  st.write("Please select an operation mode.")
