# Coder/skills_engine.py
import re
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional

class AgentSkillsEngine:
    """
    Bộ engine phân tích và vận hành thư viện kỹ năng
    tương thích hoàn toàn với tiêu chuẩn agentskills.io
    """
    def __init__(self, workspace_path: str):
        # 1. Workspace vẫn giữ để thực thi các tệp tin của người dùng
        self.workspace = Path(workspace_path).expanduser().resolve()
        
        # 2. SỬA ĐỔI QUAN TRỌNG: Định vị .skills nằm cùng cấp với file skills_engine.py (trong thư mục Coder/)
        self.skills_dir = Path(__file__).parent.resolve() / ".skills"
        
    def scan_catalog(self) -> List[Dict[str, str]]:
        """Tier 1: Quét nhanh tên và mô tả sơ bộ của toàn bộ skill hiện có."""
        catalog = []
        if not self.skills_dir.exists() or not self.skills_dir.is_dir():
            return catalog

        for skill_path in self.skills_dir.iterdir():
            if skill_path.is_dir():
                skill_md = skill_path / "SKILL.md"
                if skill_md.exists():
                    metadata = self._parse_skill_frontmatter(skill_md)
                    if metadata:
                        catalog.append(metadata)
        return catalog

    def load_skill_body(self, skill_name: str) -> Optional[str]:
        """Tier 2: Đọc toàn bộ nội dung hướng dẫn sử dụng trong file SKILL.md."""
        skill_md = self.skills_dir / skill_name / "SKILL.md"
        if not skill_md.exists():
            return None
        
        content = skill_md.read_text(encoding="utf-8")
        # Giới hạn maxsplit=2 để chỉ cắt ở biên của YAML Frontmatter đầu file,
        # bảo toàn các đường kẻ ngang "---" trong nội dung Markdown phía sau.
        parts = re.split(r'^---+\s*$', content, maxsplit=2, flags=re.MULTILINE)
        if len(parts) >= 3:
            return parts[2].strip()
        return content.strip()

    def get_script_path(self, skill_name: str, script_name: str) -> Optional[Path]:
        """Tier 3: Trả về đường dẫn tuyệt đối của script thực thi."""
        script_path = self.skills_dir / skill_name / "scripts" / script_name
        if script_path.exists() and script_path.is_file():
            return script_path
        return None

    def _parse_skill_frontmatter(self, file_path: Path) -> Optional[Dict[str, str]]:
        try:
            content = file_path.read_text(encoding="utf-8")
            parts = re.split(r'^---+\s*$', content, flags=re.MULTILINE)
            if len(parts) >= 3:
                yaml_data = yaml.safe_load(parts[1])
                return {
                    "name": yaml_data.get("name", file_path.parent.name),
                    "description": yaml_data.get("description", "")
                }
        except Exception:
            pass
        return None