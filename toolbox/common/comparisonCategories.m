function category = comparisonCategories(i)
%% Find the category name of it's number/index
switch (i)
    case 1
        category = '';
    case 2
        category = '';
    case 3
        category = 'Animal';
    case 4
        category = 'Building';
    case 5
        category = 'Plant';
    case 6
        category = 'Tool';
    case 7
        category = 'Vehicle';
    otherwise
        category = 'ERROR!';
end