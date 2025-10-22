"""
Template management for realtime voice conversion settings.
"""

import os
import yaml
import gradio as gr


class RealtimeTemplateManager:
    """Manager for realtime conversion templates"""

    def __init__(self, template_folder=None):
        """
        Initialize template manager.

        Args:
            template_folder: Path to template folder. If None, uses default.
        """
        if template_folder is None:
            # Default: root/templates/real_time
            root_dir = os.getcwd()
            template_folder = os.path.join(root_dir, "templates", "real_time")

        self.template_folder = template_folder
        self._ensure_folder_exists()

    def _ensure_folder_exists(self):
        """Create template folder if it doesn't exist"""
        os.makedirs(self.template_folder, exist_ok=True)

    def get_template_list(self):
        """Get list of available template names"""
        if not os.path.exists(self.template_folder):
            return []
        templates = [
            os.path.splitext(f)[0]
            for f in os.listdir(self.template_folder)
            if f.endswith(".yaml")
        ]
        return sorted(templates)

    def validate_template_name(self, name, existing_templates=None, current_name=None):
        """
        Validate template name and return error message if invalid.

        Args:
            name: Template name to validate
            existing_templates: List of existing template names (if None, will fetch)
            current_name: Current template name (for overwrite, to exclude from duplicate check)

        Returns:
            Error message string if invalid, None if valid
        """
        if not name or not name.strip():
            return "Template name cannot be empty."

        name = name.strip()

        # Check for forbidden characters (Windows + Unix)
        forbidden_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
        for char in forbidden_chars:
            if char in name:
                return f"Template name cannot contain forbidden character: {char}"

        # Get existing templates if not provided
        if existing_templates is None:
            existing_templates = self.get_template_list()

        # Check for duplicate names (excluding current name for overwrite)
        if current_name:
            check_list = [t for t in existing_templates if t != current_name]
        else:
            check_list = existing_templates

        if name in check_list:
            return f"Template name '{name}' already exists."

        return None

    def save_template(self, name, template_data):
        """
        Save template to YAML file.

        Args:
            name: Template name
            template_data: Dictionary containing template settings
        """
        self._ensure_folder_exists()
        template_path = os.path.join(self.template_folder, f"{name}.yaml")
        with open(template_path, 'w', encoding='utf-8') as f:
            yaml.dump(template_data, f, allow_unicode=True, sort_keys=False)

    def load_template(self, name):
        """
        Load template from YAML file.

        Args:
            name: Template name

        Returns:
            Dictionary containing template settings, or None if not found
        """
        template_path = os.path.join(self.template_folder, f"{name}.yaml")
        if not os.path.exists(template_path):
            return None
        with open(template_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def delete_template(self, name):
        """
        Delete template file.

        Args:
            name: Template name
        """
        template_path = os.path.join(self.template_folder, f"{name}.yaml")
        if os.path.exists(template_path):
            os.remove(template_path)

    def create_settings_dict(
        self,
        inp_device, inp_gain, inp_asio,
        out_device, out_gain, out_asio,
        use_mon, mon_device, mon_gain, mon_asio,
        excl_mode, vad_en,
        mdl_file, idx_file,
        atune, atune_str,
        prop_pitch, prop_pitch_thresh,
        speaker_id,
        ptch, idx_rate, vol_env, prot,
        f0_meth, hybrid_ratio,
        emb_model, emb_custom,
        chnk_size, cross_fade, extra_conv, silent_thresh,
        stable_md
    ):
        """
        Create a settings dictionary from individual parameters.

        Returns:
            Dictionary in template format
        """
        return {
            "audioTab": {
                "input": {
                    "device": inp_device or "",
                    "gain": inp_gain,
                    "asio_channel": inp_asio,
                },
                "output": {
                    "device": out_device or "",
                    "gain": out_gain,
                    "asio_channel": out_asio,
                },
                "monitor": {
                    "enabled": use_mon,
                    "device": mon_device or "",
                    "gain": mon_gain,
                    "asio_channel": mon_asio,
                },
                "exclusive_mode": excl_mode,
                "vad_enabled": vad_en,
                "stable_mode": stable_md,
            },
            "modelTab": {
                "voice": {
                    "model_path": mdl_file or "",
                    "index_path": idx_file or "",
                },
                "inference": {
                    "f0_method": f0_meth,
                    "embedder_model": emb_model,
                    "embedder_model_custom": emb_custom or "",
                    "autotune": atune,
                    "autotune_strength": atune_str,
                    "proposed_pitch": prop_pitch,
                    "proposed_pitch_threshold": prop_pitch_thresh,
                    "speaker_id": speaker_id,
                },
                "parameterValues": {
                    "pitch": ptch,
                    "index_rate": idx_rate,
                    "volume_envelope": vol_env,
                    "protect": prot,
                    "hybrid_blend_ratio": hybrid_ratio,
                },
            },
            "performanceTab": {
                "chunk_size": chnk_size,
                "crossfade_overlap_size": cross_fade,
                "extra_convert_size": extra_conv,
                "silence_threshold": silent_thresh,
            },
        }

    def extract_settings_to_gradio_updates(self, template_data):
        """
        Extract settings from template data and create Gradio update objects.

        Args:
            template_data: Dictionary containing template settings

        Returns:
            Tuple of 31 gr.update() objects for all UI components
        """
        if not template_data:
            return [gr.update()] * 31

        audio = template_data.get("audioTab", {})
        model = template_data.get("modelTab", {})
        perf = template_data.get("performanceTab", {})

        return (
            # Audio tab (13 items - added stable_mode)
            gr.update(value=audio.get("input", {}).get("device", "")),
            gr.update(value=audio.get("input", {}).get("gain", 100)),
            gr.update(value=audio.get("input", {}).get("asio_channel", -1)),
            gr.update(value=audio.get("output", {}).get("device", "")),
            gr.update(value=audio.get("output", {}).get("gain", 100)),
            gr.update(value=audio.get("output", {}).get("asio_channel", -1)),
            gr.update(value=audio.get("monitor", {}).get("enabled", False)),
            gr.update(value=audio.get("monitor", {}).get("device", "")),
            gr.update(value=audio.get("monitor", {}).get("gain", 100)),
            gr.update(value=audio.get("monitor", {}).get("asio_channel", -1)),
            gr.update(value=audio.get("exclusive_mode", True)),
            gr.update(value=audio.get("vad_enabled", True)),
            gr.update(value=audio.get("stable_mode", False)),
            # Model tab (14 items)
            gr.update(value=model.get("voice", {}).get("model_path", "")),
            gr.update(value=model.get("voice", {}).get("index_path", "")),
            gr.update(value=model.get("inference", {}).get("autotune", False)),
            gr.update(value=model.get("inference", {}).get("autotune_strength", 1)),
            gr.update(value=model.get("inference", {}).get("proposed_pitch", False)),
            gr.update(value=model.get("inference", {}).get("proposed_pitch_threshold", 155.0)),
            gr.update(value=model.get("inference", {}).get("speaker_id", 0)),
            gr.update(value=model.get("parameterValues", {}).get("pitch", 0)),
            gr.update(value=model.get("parameterValues", {}).get("index_rate", 0.75)),
            gr.update(value=model.get("parameterValues", {}).get("volume_envelope", 1)),
            gr.update(value=model.get("parameterValues", {}).get("protect", 0.5)),
            gr.update(value=model.get("inference", {}).get("f0_method", "swift")),
            gr.update(value=model.get("parameterValues", {}).get("hybrid_blend_ratio", 0.5)),
            gr.update(value=model.get("inference", {}).get("embedder_model", "contentvec")),
            gr.update(value=model.get("inference", {}).get("embedder_model_custom", "")),
            # Performance tab (4 items)
            gr.update(value=perf.get("chunk_size", 512)),
            gr.update(value=perf.get("crossfade_overlap_size", 0.05)),
            gr.update(value=perf.get("extra_convert_size", 0.5)),
            gr.update(value=perf.get("silence_threshold", -90)),
        )
