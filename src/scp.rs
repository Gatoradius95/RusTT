use anyhow::Result;

pub struct ScpScript {
    pub states: Vec<State>,
}

pub struct State {
    pub name: String,
    pub conditions: Vec<Condition>,
    pub actions: Vec<Action>,
    pub reference_scripts: Vec<ReferenceScript>,
}

#[derive(Clone)]
pub struct Condition {
    pub name: String,
    pub string_arg: Option<String>,
    pub op: CondOp,
    pub value: String,
    pub goto: Option<String>,
    pub and_next: bool,
}

#[derive(Clone, PartialEq, Debug)]
pub enum CondOp {
    Equals,
    LessThan,
    GreaterThan,
    LessOrEqual,
    GreaterOrEqual,
}

pub struct Action {
    pub name: String,
    pub params: Vec<String>,
}

#[derive(Clone)]
pub struct ReferenceScript {
    pub script: String,
    pub source: String,
    pub return_state: String,
    pub conditions: Vec<Condition>,
}

fn strip_comment(line: &str) -> &str {
    let s = line.trim();
    if let Some(p) = s.find("//") {
        s[..p].trim()
    } else if let Some(p) = s.find(';') {
        s[..p].trim()
    } else {
        s
    }
}

fn unquote(s: &str) -> String {
    let t = s.trim();
    if t.starts_with('"') && t.ends_with('"') && t.len() >= 2 {
        t[1..t.len() - 1].to_string()
    } else {
        t.to_string()
    }
}

fn parse_action_params(line: &str) -> (String, Vec<String>) {
    let s = line.trim();
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut in_quote = false;
    for ch in s.chars() {
        match ch {
            '"' => {
                in_quote = !in_quote;
                current.push(ch);
            }
            ' ' | '\t' if !in_quote => {
                if !current.is_empty() {
                    tokens.push(current.clone());
                    current.clear();
                }
            }
            _ => current.push(ch),
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    if tokens.is_empty() {
        return (String::new(), Vec::new());
    }
    let name = tokens.remove(0);
    let params: Vec<String> = tokens.into_iter().map(|t| unquote(&t)).collect();
    (name, params)
}

/// Strip `{` and `}` from a line, returning the trimmed keyword and whether
/// it had an opening brace and/or a closing brace.
fn strip_braces(s: &str) -> (&str, bool, bool) {
    let s = s.trim();
    let open = s.contains('{');
    let close = s.contains('}');
    let stripped = s.trim_start_matches('{').trim_end_matches('{')
                       .trim_start_matches('}').trim_end_matches('}')
                       .trim();
    (stripped, open, close)
}

pub fn parse(data: &str) -> Result<ScpScript> {
    let mut states = Vec::new();
    let lines: Vec<&str> = data.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let line = strip_comment(lines[i]);
        i += 1;
        if line.is_empty() {
            continue;
        }

        // state <Name> {   — only detect lines that start with "state "
        let kw = strip_braces(line).0;
        if kw.starts_with("state ") || line.trim().starts_with("state ") {
            let raw = line.trim();
            let name = raw["state ".len()..]
                .trim()
                .trim_end_matches('{')
                .trim()
                .to_string();
            let mut state = State {
                name,
                conditions: Vec::new(),
                actions: Vec::new(),
                reference_scripts: Vec::new(),
            };

            // Parse the body until we close back to depth 0.
            let mut brace_depth = 1i32;
            let mut section: Section = Section::None;
            let mut ref_script = ReferenceScript {
                script: String::new(),
                source: String::new(),
                return_state: String::new(),
                conditions: Vec::new(),
            };

            while i < lines.len() && brace_depth > 0 {
                let raw_line = strip_comment(lines[i]);
                i += 1;
                if raw_line.is_empty() {
                    continue;
                }

                let (stripped, had_open, had_close) = strip_braces(raw_line);

                if had_open {
                    brace_depth += raw_line.matches('{').count() as i32;
                }
                if had_close {
                    brace_depth -= raw_line.matches('}').count() as i32;
                    // Check if closing a ReferenceScript
                    if brace_depth <= 0 {
                        break;
                    }
                    if section == Section::RefScript && raw_line.trim() == "}" {
                        section = Section::RefScriptFields;
                        state.reference_scripts.push(ref_script.clone());
                        ref_script = ReferenceScript {
                            script: String::new(),
                            source: String::new(),
                            return_state: String::new(),
                            conditions: Vec::new(),
                        };
                        continue;
                    }
                }

                let trimmed = stripped;

                if trimmed == "ReferenceScript" || trimmed.starts_with("ReferenceScript") {
                    section = Section::RefScript;
                    ref_script = ReferenceScript {
                        script: String::new(),
                        source: String::new(),
                        return_state: String::new(),
                        conditions: Vec::new(),
                    };
                    continue;
                }

                if trimmed == "Conditions" {
                    if section == Section::RefScript {
                        let (new_i, conds) = parse_condition_block(&lines, i);
                        i = new_i;
                        brace_depth -= 1;
                        ref_script.conditions = conds;
                        continue;
                    } else {
                        section = Section::Conditions;
                        continue;
                    }
                }

                if trimmed == "Actions" {
                    section = Section::Actions;
                    continue;
                }

                // Inside a section, parse content
                match section {
                    Section::Conditions => {
                        if !trimmed.is_empty() && trimmed != "{" {
                            let cond = parse_condition(trimmed)?;
                            state.conditions.push(cond);
                        }
                    }
                    Section::Actions => {
                        if !trimmed.is_empty() && trimmed != "{" {
                            let (name, params) = parse_action_params(trimmed);
                            if !name.is_empty() {
                                state.actions.push(Action { name, params });
                            }
                        }
                    }
                    Section::RefScript => {
                        if let Some(rest) = trimmed.strip_prefix("Script=") {
                            ref_script.script = rest.trim().to_string();
                        } else if let Some(rest) = trimmed.strip_prefix("Source=") {
                            ref_script.source = rest.trim().to_string();
                        } else if let Some(rest) = trimmed.strip_prefix("ReturnState=") {
                            ref_script.return_state = rest.trim().to_string();
                        }
                    }
                    Section::RefScriptFields => {
                        // Already handled above (closing brace pushed the ref_script)
                        // After the closing brace of ReferenceScript body, we continue
                        // looking for the state's next section
                        section = Section::None;
                        // Fall through: re-evaluate this line
                        // We need to re-process stripped as a section keyword
                        if trimmed == "Conditions" {
                            section = Section::Conditions;
                        } else if trimmed == "Actions" {
                            section = Section::Actions;
                        }
                    }
                    Section::None => {}
                }
            }

            states.push(state);
        }
    }

    Ok(ScpScript { states })
}

#[derive(PartialEq)]
enum Section {
    None,
    Conditions,
    Actions,
    RefScript,
    RefScriptFields,
}

fn parse_condition_block(lines: &[&str], mut i: usize) -> (usize, Vec<Condition>) {
    let mut conditions = Vec::new();
    let mut depth = 1i32;
    while i < lines.len() && depth > 0 {
        let raw = strip_comment(lines[i]);
        i += 1;
        if raw.is_empty() {
            continue;
        }

        depth += raw.matches('{').count() as i32;
        depth -= raw.matches('}').count() as i32;
        if depth <= 0 {
            break;
        }

        let trimmed = raw.trim().trim_start_matches('{').trim_start_matches('}');
        let trimmed = trimmed.trim();
        if !trimmed.is_empty() && trimmed != "{" {
            if let Ok(cond) = parse_condition(trimmed) {
                conditions.push(cond);
            }
        }
    }
    (i, conditions)
}

fn parse_condition(s: &str) -> Result<Condition> {
    let s = s.trim();

    let (mut and_next, s) = if let Some(rest) = s.strip_prefix("if ") {
        if rest.starts_with("and ") {
            (true, rest[4..].trim())
        } else {
            (false, rest)
        }
    } else if s == "and" {
        return Ok(Condition {
            name: String::new(),
            string_arg: None,
            op: CondOp::Equals,
            value: String::new(),
            goto: None,
            and_next: true,
        });
    } else {
        (false, s)
    };

    let mut goto = None;
    let remaining = if let Some(idx) = s.find(" goto ") {
        goto = Some(s[idx + 6..].trim().to_string());
        s[..idx].trim()
    } else {
        s
    };

    let (op, value_raw, base): (CondOp, String, &str) = if let Some(idx) = remaining.find(" == ") {
        (CondOp::Equals, remaining[idx + 4..].trim().to_string(), remaining[..idx].trim())
    } else if let Some(idx) = remaining.find(" != ") {
        (CondOp::Equals, format!("!{}", remaining[idx + 4..].trim()), remaining[..idx].trim())
    } else if let Some(idx) = remaining.find(" <= ") {
        (CondOp::LessOrEqual, remaining[idx + 4..].trim().to_string(), remaining[..idx].trim())
    } else if let Some(idx) = remaining.find(" >= ") {
        (CondOp::GreaterOrEqual, remaining[idx + 4..].trim().to_string(), remaining[..idx].trim())
    } else if let Some(idx) = remaining.find(" < ") {
        (CondOp::LessThan, remaining[idx + 3..].trim().to_string(), remaining[..idx].trim())
    } else if let Some(idx) = remaining.find(" > ") {
        (CondOp::GreaterThan, remaining[idx + 3..].trim().to_string(), remaining[..idx].trim())
    } else {
        (CondOp::Equals, String::new(), remaining.trim())
    };

    let mut value = value_raw.to_string();
    if value.ends_with(" and") {
        value.truncate(value.len() - 4);
        value = value.trim().to_string();
        and_next = true;
    } else if value.ends_with(" or") {
        value.truncate(value.len() - 3);
        value = value.trim().to_string();
    }

    let (name, string_arg) = if let Some(rest) = base.strip_suffix('"') {
        if let Some(qstart) = rest.rfind('"') {
            let word = rest[..qstart].trim();
            let arg = &rest[qstart + 1..];
            (word.to_string(), Some(arg.to_string()))
        } else {
            (base.to_string(), None)
        }
    } else {
        let word = base.split_whitespace().next().unwrap_or("").to_string();
        (word, None)
    };

    Ok(Condition {
        name,
        string_arg,
        op,
        value,
        goto,
        and_next,
    })
}

pub fn parse_file(path: &str) -> Result<ScpScript> {
    let data = std::fs::read_to_string(path)?;
    parse(&data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_state() {
        let input = r#"state Base {
    Conditions {
        if IAmAPartyCharacter == 1 goto PartyUpdate
    }
    Actions {
        SetOpponent "reset"
        CanOpenDoors
    }
}
"#;
        let s = parse(input).unwrap();
        assert_eq!(s.states.len(), 1);
        assert_eq!(s.states[0].name, "Base");
        assert_eq!(s.states[0].conditions.len(), 1);
        assert_eq!(s.states[0].conditions[0].name, "IAmAPartyCharacter");
        assert_eq!(s.states[0].conditions[0].op, CondOp::Equals);
        assert_eq!(s.states[0].conditions[0].value, "1");
        assert_eq!(
            s.states[0].conditions[0].goto.as_deref(),
            Some("PartyUpdate")
        );
        assert_eq!(s.states[0].actions.len(), 2);
        assert_eq!(s.states[0].actions[0].name, "SetOpponent");
        assert_eq!(s.states[0].actions[0].params, vec!["reset"]);
    }

    #[test]
    fn parse_reference_script() {
        let input = r#"state PartyUpdate {
    ReferenceScript {
        Script=GeneralParty
        Source=Global
        ReturnState=PartyUpdate
        Conditions {
            if GotOpponent == 1 and
            if OpponentIsAThreat == 1
        }
    }
    Conditions {
        if IAmAPartyCharacter == 0 goto NonPartyStart
    }
    Actions {
        ReleaseLocator
        FollowPlayer "0.75"
    }
}
"#;
        let s = parse(input).unwrap();
        assert_eq!(s.states.len(), 1);
        let st = &s.states[0];
        assert_eq!(st.reference_scripts.len(), 1);
        assert_eq!(st.reference_scripts[0].script, "GeneralParty");
        assert_eq!(st.reference_scripts[0].source, "Global");
        assert_eq!(st.reference_scripts[0].return_state, "PartyUpdate");
        assert_eq!(st.reference_scripts[0].conditions.len(), 2);
        assert_eq!(st.reference_scripts[0].conditions[0].name, "GotOpponent");
        assert_eq!(st.reference_scripts[0].conditions[0].value, "1");
        assert!(st.reference_scripts[0].conditions[0].and_next);
    }

    #[test]
    fn parse_multiple_states() {
        let input = r#"state Base {
    Conditions {
    }
    Actions {
        FollowPath "WALK"
    }
}

state GotOpponent {
    Conditions {
        if GotOpponent == 0 goto Base
    }
    Actions {
        FollowPath "WALK"
    }
}
"#;
        let s = parse(input).unwrap();
        assert_eq!(s.states.len(), 2);
        assert_eq!(s.states[0].name, "Base");
        assert_eq!(s.states[1].name, "GotOpponent");
    }

    #[test]
    fn parse_level_scp() {
        let input = r#"state Base {
    Conditions {
    }
    Actions {
        CnxController "from=shutter_a" "to=shutter_b" "off_flag=BLOCK" "obj=garage_door" "checkvisible" "on_frames=lastframeTOlastframe" "bothways"
    }
}
"#;
        let s = parse(input).unwrap();
        assert_eq!(s.states.len(), 1);
        assert_eq!(s.states[0].actions.len(), 1);
        assert_eq!(s.states[0].actions[0].name, "CnxController");
        assert_eq!(s.states[0].actions[0].params.len(), 7);
    }

    #[test]
    fn parse_condition_with_string_value() {
        let input = r#"state Update {
    Conditions {
        if InHubArea "MAINROOM" == 1 goto Enter_MainRoom
    }
    Actions {
    }
}
"#;
        let s = parse(input).unwrap();
        assert_eq!(s.states[0].conditions[0].name, "InHubArea");
        assert_eq!(s.states[0].conditions[0].string_arg.as_deref(), Some("MAINROOM"));
        assert_eq!(s.states[0].conditions[0].value, "1");
        assert_eq!(s.states[0].conditions[0].goto.as_deref(), Some("Enter_MainRoom"));
    }
}
