-- Handle description lists
function Description(desc)
    -- Check if desc is valid and has content
    if desc and desc.content then
        local items = pandoc.List()
        for _, item in ipairs(desc.content) do
            -- Ensure term and descriptions are valid
            local term = item.term and pandoc.Strong(item.term) or pandoc.Strong(pandoc.Str("Unknown"))
            local desc_items = pandoc.List()
            for _, d in ipairs(item.descriptions or {}) do
                table.insert(desc_items, pandoc.Para(d))
            end
            table.insert(items, {term = term, descriptions = desc_items})
        end
        return pandoc.Description(items)
    end
    return desc -- Return unchanged if invalid
end

-- Handle any block to catch description lists
function Block(block)
    if block.t == "Div" then
        return pandoc.walk_block(block, { Description = Description })
    end
    return block
end

-- Handle the full document
function Pandoc(doc)
    return pandoc.walk_block(pandoc.Div(doc.blocks), { Description = Description })
end