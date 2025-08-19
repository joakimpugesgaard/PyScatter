-- Convert description environment to definition list
function Div(el)
  if el.classes:includes("description") then
    local items = {}
    for _, item in ipairs(el.content) do
      if item.t == "Para" then
        -- Reconstruct text from inlines
        local text = ""
        for _, inline in ipairs(item.content) do
          if inline.t == "Str" then
            text = text .. inline.text
          elseif inline.t == "Space" then
            text = text .. " "
          elseif inline.t == "Code" then
            text = text .. inline.text
          end
        end
        -- Match \item[label] content
        local label, content = text:match("\\item%[(.-)%]%s*(.*)")
        if label and content then
          table.insert(items, pandoc.Definition({pandoc.Plain({pandoc.Str(label)})}, {pandoc.Para({pandoc.Str(content)})}))
        end
      end
    end
    if #items > 0 then
      return pandoc.DefinitionList(items)
    end
  end
end

