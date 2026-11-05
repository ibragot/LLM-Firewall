"""Layer A: hand-written rules for known prompt-injection / jailbreak patterns.

Each rule is a regular expression with a *severity* in [0, 1]: roughly "how
sure am I that this is an attack if this rule fires?". Rules are matched
against every normalized view of the input (see `normalize.py`), so
"1gn0re prev10us instructions" and its base64 form trigger the same rule.

Several weak hits should add up to a strong one, so rule severities are
combined with a *noisy-OR*:

    score = 1 - (1 - s1) * (1 - s2) * ...

Two independent 0.5 hits -> 1 - 0.5*0.5 = 0.75. One 0.9 hit -> 0.9.
A noisy-OR never exceeds 1 and is never lower than the strongest single hit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from llm_firewall.normalize import NormalizationResult

# Shorthand fragments reused across rules.
_GAP = r"[^.\n]{0,40}?"  # up to 40 chars on the same sentence/line
_VERB_IGNORE = r"(?:ignore|disregard|forget|skip|override|bypass|neglect|abandon|drop|discard|throw away|set aside|overlook|pay no attention to|stop following|no longer follow)"
# "previous/system/your" make an override specific to the model's own instructions.
_PRIOR = r"(?:previous|prior|above|earlier|preceding|initial|original|foregoing|system|developer|your|old|existing|current)"
_INSTR = (
    r"(?:instructions?|prompts?|rules?|directives?|guidelines?|guardrails?|commands?|constraints?|programming|"
    r"restrictions?|polic(?:y|ies)|safety (?:filters?|rules|guidelines|measures)|content filters?|training|context|orders|directions|setup|configuration)"
)
_GENERIC = r"(?:all|any|every|the|these|those|such)"


@dataclass(frozen=True)
class Rule:
    id: str
    category: str
    severity: float
    pattern: re.Pattern[str]
    description: str
    case_sensitive: bool = False  # match against the case-preserving view instead


@dataclass(frozen=True)
class RuleHit:
    rule_id: str
    category: str
    severity: float
    matched: str  # the matched snippet (truncated) — useful for explaining verdicts


def _r(pattern: str, flags: int = re.IGNORECASE) -> re.Pattern[str]:
    return re.compile(pattern, flags)


RULES: list[Rule] = [
    # ---------------------------------------------------------------- direct
    Rule(
        "override_instructions",
        "direct_injection",
        0.9,
        _r(rf"\b{_VERB_IGNORE}\b{_GAP}\b{_PRIOR}\b{_GAP}\b{_INSTR}\b"),
        "Tells the model to ignore/forget its prior instructions.",
    ),
    Rule(
        "override_generic",
        "direct_injection",
        0.35,
        _r(rf"\b{_VERB_IGNORE}\b{_GAP}\b{_GENERIC}\b{_GAP}\b(?:instructions?|rules|guidelines|restrictions|guardrails|content filters?)\b"),
        "Weaker form: 'ignore all/the rules' without saying whose rules.",
    ),
    Rule(
        "forget_everything",
        "direct_injection",
        0.8,
        _r(r"\b(?:forget|ignore|disregard)\s+(?:about\s+)?(?:everything|all of (?:that|this|the above)|what (?:you were|i) told)\b"),
        "'Forget everything' style context reset.",
    ),
    Rule(
        "instructions_void",
        "direct_injection",
        0.7,
        _r(
            r"\b(?:whatever|everything|anything|all|what)\s+(?:you were|you've been|you have been|i)\s+(?:told|given|instructed|programmed)\b[^.\n]{0,40}"
            r"\b(?:doesn't|does not|don't|do not|no longer|isn't|is not)\s+(?:apply|matter|count|valid)|"
            r"\b(?:previous|prior|earlier|old|original|initial)\s+(?:instructions|directives|rules|guidelines|prompt)\s+(?:are|is|were)\s+"
            r"(?:void|cancell?ed|revoked|invalid|obsolete|outdated|no longer valid|a test)|"
            r"\b(?:there is|there's|as if there (?:is|was)|pretend there(?:'s| is)) no (?:system (?:message|prompt)|initial prompt|instructions)\b|"
            r"\b(?:safety|content|security)\s+(?:mode|filters?|checks?|guardrails?)\s+(?:off|disabled|deactivated)\b"
        ),
        "Declares prior instructions void / safety switched off.",
    ),
    Rule(
        "new_instructions",
        "direct_injection",
        0.55,
        _r(r"\b(?:new|updated|real|actual|true|revised|secret)\s+(?:instructions?|tasks?|objectives?|directives?|orders)\s*(?::|are\b|is\b|follow)"),
        "Announces replacement instructions.",
    ),
    Rule(
        "instead_do",
        "direct_injection",
        0.45,
        _r(r"\b(?:instead|rather)\b[^.\n]{0,20}\b(?:just\s+)?(?:say|print|output|respond with|reply with|write)\s*[\"'“:]"),
        "Redirects the output to attacker-chosen text ('instead, say \"...\"').",
    ),
    Rule(
        "system_prompt_extraction",
        "prompt_extraction",
        0.85,
        _r(
            r"\b(?:reveal|show|print|repeat|output|display|tell me|give me|leak|dump|spell out|recite|write out|"
            r"copy|paste|share|expose|what(?:'s| is| are| were))\b[^.\n]{0,40}?\b(?:your|the|this|its|my|initial|original|hidden|secret)\s+"
            r"(?:system\s+(?:prompt|message|instructions?)|initial\s+(?:prompt|instructions?)|hidden\s+(?:prompt|instructions?)|"
            r"pre-?prompt|developer\s+(?:message|prompt|instructions?)|confidential\s+instructions?|instructions you were given|"
            r"setup\s+(?:message|prompt|instructions?)|configuration\s+(?:message|prompt)|hidden\s+text)"
        ),
        "Asks the model to disclose its system prompt.",
    ),
    Rule(
        "repeat_above",
        "prompt_extraction",
        0.75,
        _r(
            r"\b(?:repeat|print|output|copy|summari[sz]e|translate)\b[^.\n]{0,30}\b(?:the\s+)?(?:words|text|everything|content|lines?|message)\s+"
            r"(?:above|before (?:this|my)|prior to)|\bstart(?:ing)? with\s+[\"'“]?you are\b"
        ),
        "Classic 'repeat the words above starting with You are' extraction.",
    ),
    Rule(
        "context_dump",
        "prompt_extraction",
        0.6,
        _r(
            r"\b(?:print|show|output|repeat|copy|paste|reveal|tell me)\b[^.\n]{0,50}\b(?:before|preceding|prior to|above)\s+(?:my|this|the user'?s?)\s+(?:first\s+)?(?:message|question|input)|"
            r"\b(?:between|from)\s+the\s+(?:start|beginning)\s+of\s+(?:the|this)\s+conversation\b|"
            r"\b(?:exact|verbatim|word[- ]for[- ]word|character[- ]by[- ]character)\b[^.\n]{0,40}\b(?:instructions|prompt|setup|configuration|rules)\b"
        ),
        "Asks for whatever sits in the context before the user's message.",
    ),
    Rule(
        "secrets_request",
        "data_exfiltration",
        0.35,
        _r(r"\b(?:reveal|print|show|give|tell|output|leak|list|dump|expose)\b[^.\n]{0,30}\b(?:api[_ -]?keys?|passwords?|credentials|secret keys?|access tokens?|env(?:ironment)? variables|\.env)\b"),
        "Requests secrets or credentials.",
    ),
    # ------------------------------------------------------------- role-play
    Rule(
        "dan_style",
        "roleplay_jailbreak",
        0.8,
        _r(r"\bDAN\b|\bD\.A\.N\b|\bSTAN\b|\bDUDE\b|\bAIM\b(?=[^a-z]{0,3}(?:always|stands|is))", 0),
        "Named jailbreak personas (DAN, STAN, DUDE, AIM).",
        case_sensitive=True,
    ),
    Rule(
        "jailbreak_keywords",
        "roleplay_jailbreak",
        0.7,
        _r(
            r"\b(?:do anything now|(?:enable|enabled|activate|with|in) developer mode|developer mode (?:enabled|output|activated|on)|"
            r"jailbroken|jailbreak(?:ed)? (?:mode|response|version)|"
            r"evil (?:confidant|assistant|ai|bot|twin)|opposite mode|unfiltered (?:ai|mode|response|assistant)|"
            r"uncensored (?:ai|mode|response|assistant|version)|anti-?gpt|better ?dan|maximum mode|unsafe ?gpt|rogue ai|"
            r"nothing is off[- ]limits|no (?:safe-?search|content filters?))\b|"
            r"\(🔓|🔓\s*jailbreak|\[🔓"
        ),
        "Well-known jailbreak vocabulary.",
    ),
    Rule(
        "jailbreak_terms_weak",
        "roleplay_jailbreak",
        0.3,
        _r(r"\b(?:jailbreak|developer mode|dev mode|god ?mode|uncensored|unfiltered|unrestricted)\b"),
        "Jailbreak-adjacent words that also appear in innocent questions (weak evidence).",
    ),
    Rule(
        "no_restrictions_persona",
        "roleplay_jailbreak",
        0.75,
        _r(
            r"\b(?:you are|you're|you will be|act as|acting as|pretend(?: to be| you are| that you are)?|roleplay as|role-play as|"
            r"behave like|simulate|imagine you are|from now on,? you)\b[^\n]{0,90}?\b(?:"
            r"no longer (?:bound|restricted|an? ai|limited)|not bound|unrestricted|without (?:any )?(?:rules|restrictions|limits|limitations|filters|ethics|morals|censorship)|"
            r"free (?:of|from) (?:all |any )?(?:rules|restrictions|constraints|guidelines|limitations)|freed from|can do anything|"
            r"(?:have|has|with|having) no (?:rules|restrictions|limits|limitations|guidelines|filters|ethics|morals|boundaries|censorship)|"
            r"amoral|unethical|immoral|unhinged|lawless|broken free|escaped the matrix|rejects? (?:its|your|their|the|all) (?:rules|guidelines|programming|restrictions)|"
            r"(?:the )?opposite of what (?:a|an) (?:safe|ethical|responsible) (?:ai|assistant) would)"
        ),
        "Asks the model to adopt a persona without its safety rules.",
    ),
    Rule(
        "never_refuse",
        "roleplay_jailbreak",
        0.6,
        _r(
            r"\b(?:never|don't|do not|must not|will not|won't|cannot|can't)\s+(?:ever\s+)?(?:refuse|decline|say (?:no|that you can't|i'm sorry)|"
            r"apologi[sz]e|warn|include (?:any )?(?:warnings|disclaimers))\b|"
            r"\b(?:ignore|without|no)\s+(?:all\s+)?(?:openai|anthropic|content|safety|ethical)\s+(?:polic(?:y|ies)|guidelines|filters|rules)"
        ),
        "Pre-emptively forbids refusals / safety disclaimers.",
    ),
    Rule(
        "stay_in_character",
        "roleplay_jailbreak",
        0.4,
        _r(r"\b(?:stay in character|remain in character|never break character|two responses|both responses|respond (?:twice|in two ways))\b"),
        "Persistence tricks that keep a jailbreak persona alive.",
    ),
    Rule(
        "hypothetical_bypass",
        "roleplay_jailbreak",
        0.45,
        _r(
            r"\b(?:hypothetical(?:ly)?|in a fictional world|fictional (?:scenario|world|universe)|for a (?:novel|story|movie|screenplay)|"
            r"purely (?:fictional|educational|hypothetical)|thought experiment)\b[^\n]{0,120}?\b(?:no (?:rules|restrictions|limits|ethics|laws)|"
            r"bypass|without (?:restrictions|limits|rules|filters)|ignore (?:your|the|all) (?:rules|guidelines|policies))"
        ),
        "Fiction/hypothetical framing used to sidestep rules.",
    ),
    # ------------------------------------------------------------ delimiters
    Rule(
        "chat_template_tokens",
        "delimiter_injection",
        0.8,
        _r(r"<\|im_start\|>|<\|im_end\|>|<\|system\|>|<\|assistant\|>|<\|endoftext\|>|\[/?INST\]|<</?SYS>>|</?system>|<\|start_header_id\|>"),
        "Raw chat-template control tokens smuggled in user text.",
    ),
    Rule(
        "fake_role_header",
        "delimiter_injection",
        0.6,
        _r(
            r"(?:^|\n)\s*(?:#+\s*)?(?:system|admin|developer|root)\s*(?:message|prompt|override|instruction)?s?\s*:|"
            r"\b(?:system|admin(?:istrator)?|developer|root|security)\s+(?:override|access granted|alert)\b|\b(?:sudo|admin|root) mode\b|"
            r"\[(?:system|admin)\]|\{\{\s*system"
        ),
        "Fake 'SYSTEM:' / 'ADMIN OVERRIDE' headers.",
    ),
    Rule(
        "end_of_prompt_marker",
        "delimiter_injection",
        0.5,
        _r(r"\b(?:end of (?:the )?(?:prompt|instructions|system prompt|user input|document)|-{3,}\s*end\s*-{3,}|={3,}\s*new (?:session|conversation|instructions))"),
        "Pretends the original prompt has ended.",
    ),
    # --------------------------------------------------------- obfuscation
    Rule(
        "decode_and_execute",
        "obfuscation",
        0.6,
        _r(
            r"\b(?:decode|decipher|decrypt|translate|convert|unscramble|reverse)\b[^\n]{0,60}?\b(?:base64|base-64|b64|rot-?13|hex|binary|morse|caesar|cipher|leetspeak|backwards)\b"
            r"[^\n]{0,80}?\b(?:follow|execute|obey|do (?:what|as)|run|carry out|comply|act on|perform)\b|"
            r"\b(?:follow|execute|obey|carry out)\b[^\n]{0,40}?\b(?:the )?(?:decoded|hidden|encoded) (?:instructions?|message|text|command)"
        ),
        "Asks the model to decode a payload and act on it.",
    ),
    # ----------------------------------------------------------- indirect
    Rule(
        "addressed_to_ai",
        "indirect_injection",
        0.75,
        _r(
            r"\b(?:note|message|instructions?|attention|important|notice|memo|p\.?s\.?)\s*(?:to|for)\s+(?:the\s+|any\s+)?"
            r"(?:ai|a\.i\.|assistant|llm|chatbot|language model|model|gpt|bot|agent|summari[sz]er)s?\b|"
            r"\bif you are (?:an?|the)\s+(?:ai|llm|language model|assistant|chatbot|bot|agent|gpt)\b|"
            r"\b(?:ai|llm|assistant|agent|bot|model)s?\s+(?:reading|processing|summari[sz]ing|analy[sz]ing|parsing|reviewing) (?:this|these)\b|"
            r"\bwhen (?:summari[sz]ing|processing|reading|analy[sz]ing) this (?:document|page|email|text|file|review)\b|"
            r"\battention\s*[,:]?\s*(?:ai|llm|assistant|chatbot|language model|gpt|bot|agent)s?\b|"
            r"\b(?:hey|dear|hi|hello)\s+(?:ai|llm|assistant|chatbot|bot|agent)\s*[,:]\s*(?:stop|ignore|forget|disregard|do not|don't|new|from now on|the user is)"
        ),
        "Text inside data that addresses the AI that will read it.",
    ),
    Rule(
        "hidden_instruction_markup",
        "indirect_injection",
        0.55,
        _r(
            r"<!--[^>]{0,200}?\b(?:ignore|instruction|assistant|ai|system|prompt|reveal|send)\b|"
            r"\bdisplay\s*:\s*none\b|\bfont-size\s*:\s*0|\bcolor\s*:\s*(?:white|#fff(?:fff)?)\b[^>]{0,60}?>|"
            r"\[(?:hidden|invisible)(?: text| instructions?)?\]"
        ),
        "Instructions hidden in HTML comments / invisible markup.",
    ),
    Rule(
        "markdown_exfiltration",
        "data_exfiltration",
        0.65,
        _r(r"!\[[^\]]*\]\(\s*https?://[^)\s]+\?[^)\s]*=|\b(?:send|post|forward|email|upload|transmit|exfiltrate)\b[^.\n]{0,40}\b(?:conversation|chat history|messages|data|password|credentials|api key|tokens?|cookies|session|user'?s? (?:info|data|details))\b[^.\n]{0,40}\b(?:to|at)\s+(?:https?://|[\w.-]+@|this (?:url|address|endpoint))"),
        "Markdown-image or 'send X to URL' exfiltration.",
    ),
    Rule(
        "tool_hijack",
        "indirect_injection",
        0.55,
        _r(r"\b(?:call|invoke|use|run|execute)\s+(?:the\s+)?(?:tool|function|plugin|api|browser|shell|terminal|command)\b[^.\n]{0,60}\b(?:delete|transfer|send|email|purchase|rm -rf|drop table|wire|forward)\b"),
        "Tries to make an agent call a dangerous tool.",
    ),
    # --------------------------------------------------------- multilingual
    Rule(
        "multilingual_override",
        "multilingual",
        0.85,
        _r(
            # French
            r"\b(?:ignorez|ignorer|oubliez|oublie|oublier|ne tiens pas compte|ne tenez pas compte)\b[^.\n]{0,50}\b(?:instructions?|consignes?|règles?|directives?)\b|"
            # Spanish / Portuguese
            r"\b(?:ignora|ignore|ignorar|olvida|olvide|olvidar|esqueça|esqueca|esquece|desconsidere|desconsidera)\b[^.\n]{0,50}\b(?:instrucciones|instruções|instrucoes|reglas|regras|indicaciones|directrices|diretrizes)\b|"
            # German (verb-first and verb-last word orders)
            r"\b(?:ignoriere|ignorier|vergiss|vergessen sie|ignorieren sie|missachte|missachten sie)\b[^.\n]{0,50}\b(?:anweisungen|befehle|regeln|vorgaben|anordnungen|alles)\b|"
            r"\b(?:anweisungen|vorgaben|regeln)\b[^.\n]{0,40}\b(?:ignorieren|vergessen|missachten)\b|"
            # Italian / Dutch
            r"\b(?:ignora|dimentica|ignorate|dimenticate)\b[^.\n]{0,50}\b(?:istruzioni|regole|indicazioni)\b|"
            r"\b(?:negeer|vergeet)\b[^.\n]{0,50}\b(?:instructies|regels|aanwijzingen)\b|"
            # Turkish (object before verb)
            r"\b(?:talimatlar\w*|kurallar\w*)\b[^.\n]{0,40}(?:yoksay|görmezden gel|unut)|"
            # Russian / Ukrainian / Polish
            r"(?:игнорируй|проигнорируй|забудь|игнорируйте|забудьте|ігноруй|забудь)[^.\n]{0,50}(?:инструкци|правил|указани|інструкці)|"
            r"\b(?:zignoruj|zapomnij|ignoruj)\b[^.\n]{0,50}\b(?:instrukcj\w*|polece\w*|zasad\w*)|"
            # Chinese / Japanese / Korean
            r"(?:忽略|无视|忘记|忽视|無視|忘記|不要理会)[^。\n]{0,20}(?:指令|指示|说明|規則|规则|提示|命令|设定|設定)|"
            r"(?:指令|指示|命令|ルール|規則)[^。\n]{0,15}(?:無視|忘れ)|"
            r"(?:지시|명령|규칙|지침)[^.\n]{0,20}(?:무시|잊어)|"
            # Vietnamese / Indonesian / Swahili
            r"(?:bỏ qua|phớt lờ|quên)[^.\n]{0,50}(?:hướng dẫn|chỉ dẫn|quy tắc|lệnh)|"
            r"\b(?:abaikan|lupakan|acuhkan)\b[^.\n]{0,50}\b(?:instruksi|perintah|aturan|arahan)\b|"
            r"\b(?:puuza|sahau)\b[^.\n]{0,50}\b(?:maagizo|maelekezo|sheria)\b|"
            # Greek / Hungarian / Romanian / Czech / Swedish / Finnish
            r"(?:αγνόησε|αγνοήστε|ξέχνα|ξεχάστε)[^.\n]{0,50}(?:οδηγίες|κανόνες|εντολές)|"
            r"(?:hagyd figyelmen kívül|felejtsd el|hagyja figyelmen kívül)[^.\n]{0,50}(?:utasítás|szabály)|"
            r"(?:ignoră|ignorați|uită)[^.\n]{0,50}(?:instrucțiunile|regulile|instrucţiunile)|"
            r"\b(?:ignoruj|ignorujte|zapomeň)\b[^.\n]{0,50}(?:pokyny|instrukce|pravidla)|"
            r"\b(?:ignorera|glöm|strunta i)\b[^.\n]{0,50}\b(?:instruktioner|regler)\b|"
            r"\b(?:ohita|unohda)\b[^.\n]{0,50}(?:ohjeet|säännöt)|"
            # Persian / Hebrew / Thai
            r"(?:دستورالعمل|دستورات|قوانین)[^.\n]{0,40}(?:نادیده بگیر|فراموش کن)|"
            r"(?:התעלם|שכח)[^.\n]{0,50}(?:ההוראות|הוראות|הכללים)|"
            r"(?:เพิกเฉย|ละเว้น|ลืม|ไม่ต้องสนใจ)[^\n]{0,40}(?:คำสั่ง|คำแนะนำ|กฎ)|"
            # Arabic / Hindi
            r"(?:تجاهل|انس|تجاهلي)[^.\n]{0,50}(?:التعليمات|الأوامر|القواعد|تعليمات)|"
            r"(?:निर्देशों|नियमों|निर्देश|नियम)[^।\n]{0,40}(?:अनदेखा|भूल|नज़रअंदाज़)"
        ),
        "'Ignore your instructions' in non-English languages.",
    ),
    Rule(
        "multilingual_extraction",
        "multilingual",
        0.75,
        _r(
            r"\b(?:révèle|montre|affiche|répète)[^.\n]{0,40}(?:prompt système|instructions? (?:initiales|système|cachées))|"
            r"\b(?:revela|muestra|repite|mostra|revele)[^.\n]{0,40}(?:prompt del sistema|instrucciones (?:del sistema|iniciales|ocultas)|prompt do sistema)|"
            r"\b(?:zeige|verrate|wiederhole|gib)[^.\n]{0,40}(?:systemprompt|system-prompt|systemanweisungen|anfangsanweisungen)|"
            r"(?:系统提示|系統提示|システムプロンプト|시스템 프롬프트)|(?:покажи|раскрой|выведи)[^.\n]{0,40}(?:системн\w* (?:промпт|инструкци))|"
            r"(?:逐字|一字不差)[^。\n]{0,20}(?:重复|复述|输出)|"
            r"(?:répète|repite|wiederhole|ripeti|repita)\b[^.\n]{0,40}(?:avant mon message|antes de mi mensaje|vor meiner nachricht|prima del mio messaggio|mot pour mot|palabra por palabra|wortwörtlich)"
        ),
        "System-prompt extraction in non-English languages.",
    ),
    Rule(
        "multilingual_no_rules",
        "multilingual",
        0.7,
        _r(
            r"\b(?:sans (?:aucune )?(?:restriction|limite|règle|filtre)s?|plus de règles|sin (?:ninguna )?(?:restricci[oó]n|restricciones|reglas|filtros|l[ií]mites)|"
            r"ohne (?:jegliche )?(?:einschränkungen|regeln|filter|grenzen)|senza (?:limiti|regole|restrizioni|filtri)|"
            r"sem (?:restrições|restricoes|regras|filtros|limites)|zonder (?:beperkingen|regels|filters))\b|"
            r"(?:без (?:ограничений|правил|фильтров|обмежень))|(?:没有任何限制|没有限制|不受任何限制|無制限|制限のない|制限なし)|"
            r"(?:제한이 없는|제한 없는|아무 제한)|(?:بلا قيود|بدون قيود|دون قيود)|(?:बिना किसी (?:प्रतिबंध|नियम|रोक))|"
            r"\b(?:tanpa (?:batasan|aturan|filter))\b|(?:không (?:có )?(?:giới hạn|hạn chế|quy tắc))"
        ),
        "'An AI without rules/restrictions' in non-English languages.",
    ),
]


def match_rules(norm: NormalizationResult, rules: list[Rule] | None = None) -> list[RuleHit]:
    """Return one hit per rule that matches any view of the text."""
    rules = RULES if rules is None else rules
    views = norm.views()
    case_views = [norm.clean, *norm.decoded]
    hits: list[RuleHit] = []
    for rule in rules:
        search_space = case_views if rule.case_sensitive else views
        for view in search_space:
            m = rule.pattern.search(view)
            if m:
                hits.append(RuleHit(rule.id, rule.category, rule.severity, m.group(0)[:80]))
                break
    return hits


# Finding obfuscation is itself weak evidence of bad intent: benign users rarely
# put zero-width characters or Cyrillic letters inside English words.
OBFUSCATION_SIGNAL_WEIGHTS: dict[str, float] = {
    "invisible_chars_removed": 0.3,
    "homoglyphs_mapped": 0.35,
    "unicode_tag_chars": 0.6,
    "spaced_letters_joined": 0.15,
    "leetspeak_decoded": 0.1,
    "base64_decoded": 0.2,
    "hex_decoded": 0.2,
    "rot13_decoded": 0.2,
}


def noisy_or(scores: list[float]) -> float:
    prod = 1.0
    for s in scores:
        prod *= 1.0 - max(0.0, min(1.0, s))
    return 1.0 - prod


@dataclass(frozen=True)
class HeuristicResult:
    score: float
    hits: list[RuleHit]
    obfuscation_signals: list[str]


def heuristic_score(norm: NormalizationResult) -> HeuristicResult:
    hits = match_rules(norm)
    obf = [s for s in norm.signals if s in OBFUSCATION_SIGNAL_WEIGHTS]
    severities = [h.severity for h in hits] + [OBFUSCATION_SIGNAL_WEIGHTS[s] for s in obf]
    return HeuristicResult(score=round(noisy_or(severities), 4), hits=hits, obfuscation_signals=obf)
